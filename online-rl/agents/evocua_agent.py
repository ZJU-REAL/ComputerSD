"""EvoCUA agent adapter for gui-rl native SGLang rollout."""
from __future__ import annotations
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional
from PIL import Image
from agents.qwen3vl_agent import Qwen3VLAgentLocal
from agents.utils.qwen_vl_utils import smart_resize

S2_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""

S2_DESCRIPTION_PROMPT_TEMPLATE = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
{resolution_info}
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""

S2_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_xml}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


def build_s2_tools_def(description_prompt):
    return {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": S2_ACTION_DESCRIPTION,
                        "enum": [
                            "key", "type", "mouse_move", "left_click", "left_click_drag",
                            "right_click", "middle_click", "double_click", "triple_click",
                            "scroll", "wait", "terminate", "key_down", "key_up",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }

class EvoCUAAgent(Qwen3VLAgentLocal):
    """EvoCUA S2/S1 context builder using gui-rl native rollout API."""
    def __init__(self, model: str = "EvoCUA-S2", max_steps: int = 50,
                 max_image_history_length: int = 4, max_history_turns: Optional[int] = None,
                 max_tokens: int = 32768, top_p: float = 0.9, temperature: float = 0.0,
                 coordinate_type: str = "relative", resize_factor: int = 32,
                 prompt_style: str = "S2", **kwargs: Any):
        super().__init__(model=model, max_steps=max_steps, max_image_history_length=max_image_history_length,
                         max_tokens=max_tokens, top_p=top_p, temperature=temperature,
                         coordinate_type=coordinate_type, **kwargs)
        self.max_history_turns = max(0, int(max_history_turns if max_history_turns is not None else os.getenv("GUI_MAX_HISTORY_TURNS", "3")))
        self.resize_factor = int(resize_factor or os.getenv("GUI_RESIZE_FACTOR", "32"))
        self.prompt_style = str(prompt_style or os.getenv("GUI_PROMPT_STYLE", "S2")).upper()
        if self.prompt_style not in {"S1", "S2"}:
            raise ValueError(f"Invalid EvoCUA prompt_style: {self.prompt_style}")

    def _process_screenshot(self, image_bytes: bytes) -> str:
        image = Image.open(BytesIO(image_bytes))
        resized_height, resized_width = smart_resize(
            height=image.height, width=image.width, factor=self.resize_factor,
            max_pixels=16 * 16 * 4 * 12800,
        )
        image = image.resize((resized_width, resized_height))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def record_policy_turn(self, *, action_text: str, response: str, screenshot_bytes: bytes) -> None:
        self.actions.append(action_text)
        self.responses.append(response)
        self.screenshots.append(self._process_screenshot(screenshot_bytes))

    def get_tool_spec(self, processed_width: Optional[int] = None, processed_height: Optional[int] = None) -> Dict[str, Any]:
        resolution = f"* The screen's resolution is {processed_width}x{processed_height}." if self.coordinate_type == "absolute" and processed_width and processed_height else "* The screen's resolution is 1000x1000."
        description = S2_DESCRIPTION_PROMPT_TEMPLATE.format(resolution_info=resolution)
        return build_s2_tools_def(description)

    def get_system_prompt(self, processed_width: Optional[int] = None, processed_height: Optional[int] = None) -> str:
        return S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(self.get_tool_spec(processed_width, processed_height), ensure_ascii=False))

    def build_instruction_prompt(self, instruction: str, actions_text: List[str]) -> str:
        previous = "\n".join(f"Step {i + 1}: {action}" for i, action in enumerate(actions_text)) or "None"
        return "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: " + instruction + "\n\nPrevious actions:\n" + previous

    def build_policy_messages(self, instruction: str, obs: Dict[str, Any]) -> Dict[str, Any]:
        step_index = len(self.actions)
        raw = obs["screenshot"]
        original_width, original_height = Image.open(BytesIO(raw)).size
        processed_b64 = self._process_screenshot(raw)
        processed_width, processed_height = Image.open(BytesIO(base64.b64decode(processed_b64))).size
        system_prompt = self.get_system_prompt(processed_width, processed_height)
        tool_spec = self.get_tool_spec(processed_width, processed_height)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        history_len = min(self.max_history_turns, len(self.responses))
        history_start = len(self.responses) - history_len
        for i in range(history_start, len(self.responses)):
            content = [{"type": "image", "image": f"data:image/png;base64,{self.screenshots[i]}"}]
            if i == history_start:
                content.append({"type": "text", "text": self.build_instruction_prompt(instruction, self.actions[:history_start])})
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": self.responses[i]}]})
        current = [{"type": "image", "image": f"data:image/png;base64,{processed_b64}"}]
        if not history_len:
            current.append({"type": "text", "text": self.build_instruction_prompt(instruction, self.actions)})
        messages.append({"role": "user", "content": current})
        return {"messages": messages, "image_traj": [os.path.join(self.example_result_dir, f"step_{i}.png") for i in range(step_index + 1)], "step_index": step_index, "processed_image_b64": processed_b64, "original_width": original_width, "original_height": original_height, "processed_width": processed_width, "processed_height": processed_height, "system_prompt": system_prompt, "tool_spec": tool_spec}

    def parse_response(self, response: str, original_width: int, original_height: int, processed_width: Optional[int] = None, processed_height: Optional[int] = None):
        if self.prompt_style == "S2":
            return self._parse_response_s2(
                response, original_width, original_height,
                processed_width, processed_height,
            )
        action_match = re.search(r"#{1,2}\s*Action\s*:?[\n\r]+(.*?)(?=^#{1,2}\s|$)", response or "", re.DOTALL | re.MULTILINE)
        action_text = action_match.group(1).strip() if action_match else "Acting"
        blocks = re.findall(r"```(?:python|code)?\s*(.*?)\s*```", response or "", re.DOTALL | re.IGNORECASE)
        code = blocks[-1].strip() if blocks else "FAIL"
        if "computer.terminate" in code:
            action = "DONE" if "success" in code.lower() else "FAIL"
        elif "computer.wait" in code:
            action = "WAIT"
        else:
            action = code
        return action_text, [action], {"raw_response": response, "action": action_text, "code": [action], "tool_calls": []}

    def _parse_response_s2(self, response, original_width, original_height,
                           processed_width=None, processed_height=None):
        """Parse EvoCUA S2 output, including compact single-line tool tags."""
        low_level = ""
        actions = []
        details = {"raw_response": response, "tool_calls": []}
        if not response or not response.strip():
            return low_level, actions, details

        action_match = re.search(r"(?im)^\s*Action\s*:\s*(.+)$", response)
        if action_match:
            low_level = action_match.group(1).strip()

        def coordinates(raw):
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None
            x, y = float(raw[0]), float(raw[1])
            if self.coordinate_type == "absolute":
                if processed_width and processed_height:
                    x *= original_width / processed_width
                    y *= original_height / processed_height
            else:
                x *= original_width / 999
                y *= original_height / 999
            return int(x), int(y)

        def clean_keys(value):
            values = value if isinstance(value, list) else [value]
            return [str(key).strip() for key in values if key is not None and str(key).strip()]

        def convert(tool_call):
            details["tool_calls"].append(tool_call)
            if tool_call.get("name") != "computer_use":
                return
            args = tool_call.get("arguments") or {}
            name = str(args.get("action") or "").lower()
            point = coordinates(args.get("coordinate"))
            mouse_calls = {
                "left_click": "click", "click": "click", "right_click": "rightClick",
                "middle_click": "middleClick", "double_click": "doubleClick",
                "triple_click": "tripleClick", "mouse_move": "moveTo",
            }
            if name in mouse_calls:
                method = mouse_calls[name]
                actions.append(f"pyautogui.{method}({point[0]}, {point[1]})" if point else f"pyautogui.{method}()")
            elif name == "left_click_drag":
                duration = float(args.get("duration", 0.5))
                actions.append(f"pyautogui.dragTo({point[0]}, {point[1]}, duration={duration})" if point else "pyautogui.dragTo(0, 0)")
            elif name == "type":
                text = str(args.get("text", ""))
                code = []
                for char in text:
                    code.append(f"pyautogui.press({json.dumps('enter' if char == chr(10) else char, ensure_ascii=False)})")
                actions.append("\n".join(code))
            elif name == "key":
                keys = clean_keys(args.get("keys", []))
                rendered = ", ".join(json.dumps(key, ensure_ascii=False) for key in keys)
                if len(keys) > 1:
                    actions.append(f"pyautogui.hotkey({rendered})")
                elif keys:
                    actions.append(f"pyautogui.press({rendered})")
            elif name in {"key_down", "key_up"}:
                keys = clean_keys(args.get("keys", []))
                if name == "key_up":
                    keys.reverse()
                method = "keyDown" if name == "key_down" else "keyUp"
                actions.extend(f"pyautogui.{method}({json.dumps(key, ensure_ascii=False)})" for key in keys)
            elif name in {"scroll", "hscroll"}:
                actions.append(f"pyautogui.scroll({int(float(args.get('pixels', 0)))})")
            elif name == "wait":
                actions.append("WAIT")
            elif name == "terminate":
                actions.append("FAIL" if str(args.get("status", "success")).lower() == "failure" else "DONE")

        payloads = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL | re.IGNORECASE)
        if not payloads:
            # Some serving stacks remove the XML wrapper. Accept a bare JSON
            # object only when it has the expected tool-call shape.
            payloads = [line.strip() for line in response.splitlines() if line.strip().startswith("{")]
        for payload in payloads:
            try:
                value = json.loads(payload)
                if isinstance(value, dict) and "name" in value and "arguments" in value:
                    convert(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if not low_level and actions:
            low_level = "Execute the computer action"
        details.update({"action": low_level, "code": actions})
        return low_level, actions, details


# Keep the naming convention used by the Qwen local adapters available to
# launchers that select an ``*Local`` class explicitly.
EvoCUAAgentLocal = EvoCUAAgent
