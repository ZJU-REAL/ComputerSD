"""MobileWorld policy agent — subclasses the OSWorld Qwen3-VL agent.

The only world-specific surface is the prompt and the parser:
  - get_system_prompt  -> MobileWorld's official qwen3vl system prompt
  - build_instruction_prompt -> the official per-turn user template
  - parse_response     -> model output -> cluster device action dicts
  - get_tool_spec      -> the mobile tool (kept consistent in snapshots; unused
                          by tokenization, which never passes tools=)

Everything else — multimodal packing, sglang generate, dynamic-history training
data, the image-history window, and the relative coordinate machinery — is
inherited verbatim from Qwen3VLAgentLocal.

Action grammar mirrors MobileWorld's reference qwen3vl agent (mobile_use tool,
999x999 grid). Coordinates are emitted on the 0..999 grid by the model and
mapped to absolute device pixels here (== original screenshot resolution), which
is exactly what the env's `adb input tap` expects (no further scaling).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.qwen3vl_agent import Qwen3VLAgentLocal
from agents.prompts.qwen3vl_mobile import (
    MOBILE_QWEN3VL_SYSTEM_PROMPT,
    MOBILE_QWEN3VL_USER_TEMPLATE,
    MOBILE_USE_TOOL,
)

logger = logging.getLogger("desktopenv.qwen3vl_mobile_agent_local")


class Qwen3VLMobileAgentLocal(Qwen3VLAgentLocal):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("platform", "android")
        # The base asserts action_space == "pyautogui"; the mobile parser never
        # reads action_space, so force a valid value to satisfy the assert.
        kwargs["action_space"] = "pyautogui"
        super().__init__(**kwargs)  # coordinate_type flows from GUI_COORDINATE_TYPE (relative)

    # --- world-specific surface: prompt ---
    def get_system_prompt(self, processed_width: Optional[int] = None, processed_height: Optional[int] = None) -> str:
        return MOBILE_QWEN3VL_SYSTEM_PROMPT

    def get_tool_spec(self, processed_width: Optional[int] = None, processed_height: Optional[int] = None) -> Dict[str, Any]:
        return MOBILE_USE_TOOL  # unused by tokenization (no tools= passed); kept for snapshot consistency

    def build_instruction_prompt(self, instruction: str, actions_text: List[str]) -> str:
        # Match the official agent's step formatting (implementations/qwen3vl.py):
        # "Step i: <text>; " per turn (trailing "; "; "" when no steps), newlines
        # and quotes stripped from each step's text.
        steps = "".join(
            "Step {}: {}; ".format(i + 1, str(a).replace("\n", "").replace('"', ""))
            for i, a in enumerate(actions_text)
        )
        return MOBILE_QWEN3VL_USER_TEMPLATE.format(instruction=instruction, steps=steps)

    # --- world-specific surface: parser (model output -> cluster device actions) ---
    def parse_response(
        self, response: str, original_width: int, original_height: int,
        processed_width: Optional[int] = None, processed_height: Optional[int] = None,
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        low_level_instruction = ""
        actions: List[Any] = []                       # device dicts and/or "DONE"/"FAIL"/"WAIT"
        other: Dict[str, Any] = {"raw_response": response, "tool_calls": []}
        if response is None or not response.strip():
            return low_level_instruction, actions, other

        def to_px(coord: List[float]) -> Tuple[int, int]:
            # accept [x, y] or [x1, y1, x2, y2] (box -> center); 0..999 grid -> device pixels
            if len(coord) == 4:
                cx, cy = (coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2
            else:
                cx, cy = coord[0], coord[1]
            return int(float(cx) * original_width / 999), int(float(cy) * original_height / 999)

        def device(action_type: str, **payload: Any) -> Dict[str, Any]:
            return {"kind": "device", "type": action_type, "payload": payload}

        def process_tool_call(json_str: str) -> None:
            try:
                tc = json.loads(json_str)
                other["tool_calls"].append(tc)
                if tc.get("name") != "mobile_use":
                    return
                a = tc.get("arguments", {})
                action = a.get("action")
                if action == "click":
                    x, y = to_px(a["coordinate"]); actions.append(device("click", x=x, y=y))
                elif action == "long_press":
                    x, y = to_px(a["coordinate"]); actions.append(device("long_press", x=x, y=y))
                elif action == "swipe":
                    sx, sy = to_px(a["coordinate"]); ex, ey = to_px(a["coordinate2"])
                    actions.append(device("drag", start_x=sx, start_y=sy, end_x=ex, end_y=ey))
                elif action == "type":
                    actions.append(device("input_text", text=str(a.get("text", ""))))
                elif action == "answer":
                    actions.append(device("answer", text=str(a.get("text", ""))))
                    actions.append("DONE")            # answer is non-terminal at the adapter; force end
                elif action == "system_button":
                    native = {"back": "navigate_back", "home": "navigate_home", "enter": "keyboard_enter"}.get(
                        str(a.get("button", "")).lower()
                    )
                    if native:
                        actions.append(device(native))
                    else:  # "Menu" has no cluster device action
                        logger.warning("Unsupported system_button %r; skipping", a.get("button"))
                elif action == "ask_user":
                    # No human in the RL loop (interaction tasks are filtered out); treat a stray
                    # ask_user as a benign wait so it never errors the env.
                    actions.append("WAIT")
                elif action == "wait":
                    actions.append("WAIT")
                elif action == "terminate":
                    actions.append("DONE" if (a.get("status") or "success").lower() == "success" else "FAIL")
                else:
                    logger.warning("Unknown mobile action %r; skipping", action)
            except Exception as e:
                logger.error(f"Failed to parse mobile tool call: {e}")

        # ---- text scan: same scaffolding as Qwen3VLAgentLocal.parse_response ----
        inside_tool_call, current = False, []
        for raw in response.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("action:"):
                if not low_level_instruction:
                    low_level_instruction = line.split(":", 1)[-1].strip()
                continue
            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue
            if line.startswith("</tool_call>"):
                inside_tool_call = False
                if current:
                    process_tool_call("\n".join(current))
                    current = []
                continue
            if inside_tool_call:
                current.append(line)
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if "name" in obj and "arguments" in obj:
                        process_tool_call(line)
                except Exception:
                    pass
        if current:
            process_tool_call("\n".join(current))

        if not low_level_instruction and actions:
            low_level_instruction = "Execute the tool call"
        other["action"], other["code"] = low_level_instruction, actions
        return low_level_instruction, actions, other
