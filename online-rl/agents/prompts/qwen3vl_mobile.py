"""MobileWorld Qwen3-VL agent prompt — copied from MobileWorld's official agent.

This is ``MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER`` — the prompt the official qwen3vl
evaluation actually uses (mobile_world/agents/implementations/qwen3vl.py imports
and renders only this variant, unconditionally, even for GUI-only tasks). We
mirror it verbatim so the RL policy speaks the exact format the base checkpoint
was tuned/evaluated with (cold-start fidelity).

The ``<tools>`` line is rebuilt with ``json.dumps`` of the tool spec (rather than
pasted) so its escaping is byte-equivalent to the official single-line JSON
without hand-escaping ``\\n``. MCP tools are omitted (we don't use MCP); the
official template renders any MCP tools into the same ``<tools>`` block.
"""
import json

MOBILE_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": (
            "Use a touchscreen to interact with a mobile device, and take screenshots.\n"
            "* This is an interface to a mobile device with touchscreen. You can perform "
            "actions like clicking, typing, swiping, etc.\n"
            "* Some applications may take time to start or process actions, so you may need "
            "to wait and take successive screenshots to see the results of your actions.\n"
            "* The screen's resolution is 999x999.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
            "center of the element. Don't click boxes on their edges unless asked."
        ),
        "parameters": {
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `click`: Click the point on the screen with coordinate (x, y).\n"
                        "* `long_press`: Press the point on the screen with coordinate (x, y) "
                        "for specified seconds.\n"
                        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the "
                        "end point with coordinates2 (x2, y2).\n"
                        "* `type`: Input the specified text into the activated input box.\n"
                        "* `answer`: Output the answer.\n"
                        "* `system_button`: Press the system button.\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status.\n"
                        "* `ask_user`: Ask user for clarification."
                    ),
                    "enum": [
                        "click", "long_press", "swipe", "type", "answer",
                        "system_button", "wait", "ask_user", "terminate",
                    ],
                    "type": "string",
                },
                "coordinate": {
                    "description": (
                        "(x, y): The x (pixels from the left edge) and y (pixels from the top "
                        "edge) coordinates to move the mouse to. Required only by `action=click`, "
                        "`action=long_press`, and `action=swipe`."
                    ),
                    "type": "array",
                },
                "coordinate2": {
                    "description": (
                        "(x, y): The x (pixels from the left edge) and y (pixels from the top "
                        "edge) coordinates to move the mouse to. Required only by `action=swipe`."
                    ),
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type`, `action=ask_user` and `action=answer`.",
                    "type": "string",
                },
                "time": {
                    "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                    "type": "number",
                },
                "button": {
                    "description": (
                        "Back means returning to the previous interface, Home means returning to "
                        "the desktop, Menu means opening the application background menu, and Enter "
                        "means pressing the enter. Required only by `action=system_button`"
                    ),
                    "enum": ["Back", "Home", "Menu", "Enter"],
                    "type": "string",
                },
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
    },
}

MOBILE_QWEN3VL_SYSTEM_PROMPT = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    + json.dumps(MOBILE_USE_TOOL)
    + "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "# Response format\n\n"
    "Response format for every step:\n"
    "1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n"
    "2) Action: a short imperative describing what to do.\n"
    "3) A single <tool_call>...</tool_call> block containing only the JSON: "
    '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
    "Rules:\n"
    "- Output exactly in the order: Thought, Action, <tool_call>.\n"
    "- Be brief: one sentence for Thought, one for Action.\n"
    "- Do not output anything else outside those three parts.\n"
    "- If finishing, use mobile_use with action=terminate in the tool call."
)

MOBILE_QWEN3VL_USER_TEMPLATE = (
    "\nThe user query: {instruction}\n"
    "Task progress (You have done the following operation on the current device): {steps}\n"
)
