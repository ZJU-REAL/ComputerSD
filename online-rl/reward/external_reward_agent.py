import asyncio
import base64
import copy
import logging
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image

from agents.utils.qwen_vl_utils import smart_resize

logger = logging.getLogger(__name__)

# One rate limiter and HTTP client per worker event loop.  GUI rollout workers
# run many trajectories concurrently in the same loop; per-trajectory limiters
# would multiply PRM_API_MAX_CONCURRENCY by the number of trajectories.
_SHARED_RESOURCES: dict[asyncio.AbstractEventLoop, dict[str, Any]] = {}

_ANSI_GREEN = "\033[92m"
_ANSI_RESET = "\033[0m"


REWARD_SYSTEM_PROMPT = """
You are an evaluator for the most recent step of a GUI agent.
You are provided with:
1) the interaction history between the agent and the environment,
2) the agent's objective, and
3) the agent's most recent step to evaluate.
""".strip()


REWARD_INSTRUCTION_TEMPLATE = r"""
You are a strict evaluator to evaluate the most recent step of the agent in the following.

Objective of Agent:
{instruction}

Agent's most recent step (reasoning + action):
{response}
"""


REWARD_INSTRUCTION_POST = r"""
Evaluate ONLY the single most recent step using the information above.

Use the next observation AFTER executing this step (i.e., the environment state after this action) to judge whether the action actually took effect.

Assign a score of +1 if ALL of the following are true:
- The step is clearly relevant to the stated objective;
- The action is executable and coherent given the next observation;
- The next observation shows concrete progress toward the objective, not just a no-op.

Otherwise assign a score of -1, for example if the step is incorrect, irrelevant, impossible in context, has no visible effect,
undoes progress, contradicts the next observation, or hallucinates tools/objects/facts.

Think carefully but DO NOT over-think, then provide your reasoning and put the final score in \boxed{}.
"""

REWARD_GUIDANCE_POST = r"""
In addition, output one concise, actionable instruction for the actor between
[GUIDANCE_START] and [GUIDANCE_END]. The guidance must say what the actor should
do at this step or what mistake it should avoid. It must not mention that it is
a reward-model answer.
"""








def _process_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    w, h = img.size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )
    img = img.resize((resized_w, resized_h))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


from reward.prm_utils import (
    extract_prm_guidance_from_text,
    extract_prm_sign_from_text,
    read_file_bytes,
    select_prm_guidance,
)


class RewardAgent:
    def __init__(
        self,
        max_reward_image_history_length: int = 1,
        example_result_dir: str | None = None,
        tokenizer: Any | None = None,
        processor: Any | None = None,
    ):
        self.max_reward_image_history_length = max(1, int(max_reward_image_history_length))
        self.example_result_dir = example_result_dir or os.getcwd()
        self.reward_trajectory: list[dict[str, Any]] = []
        self._resource_loop: asyncio.AbstractEventLoop | None = None

        self.api_base = os.environ.get("PRM_API_BASE", "https://api.openai.com/v1")
        self.api_key = os.environ.get("PRM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "PRM_API_KEY is missing in the rollout worker. Export it before starting "
                "Ray on every node; it is intentionally not stored in Ray runtime_env."
            )
        self.api_model = os.environ.get("PRM_API_MODEL", "gpt-4o")
        self.max_concurrency = int(os.environ.get("PRM_API_MAX_CONCURRENCY", "16"))
        self.temperature = float(os.environ.get("PRM_API_TEMPERATURE", "0.6"))
        self.max_tokens = int(os.environ.get("PRM_API_MAX_TOKENS", "4096"))

    def _get_shared_resources(self) -> dict[str, Any]:
        if self._resource_loop is None:
            loop = asyncio.get_running_loop()
            state = _SHARED_RESOURCES.get(loop)
            if state is None:
                state = {
                    "semaphore": asyncio.Semaphore(self.max_concurrency),
                    "max_concurrency": self.max_concurrency,
                    "client": None,
                    "users": 0,
                }
                _SHARED_RESOURCES[loop] = state
            elif int(state["max_concurrency"]) != self.max_concurrency:
                logger.warning(
                    "PRM shared limiter already uses concurrency=%s; ignoring per-agent value=%s",
                    state["max_concurrency"], self.max_concurrency,
                )
            state["users"] += 1
            self._resource_loop = loop
        return _SHARED_RESOURCES[self._resource_loop]

    def _get_semaphore(self) -> asyncio.Semaphore:
        return self._get_shared_resources()["semaphore"]

    async def _get_client(self):
        state = self._get_shared_resources()
        if state["client"] is None:
            import httpx
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        return state["client"]

    async def aclose(self) -> None:
        """Release this trajectory's reference to the loop-shared PRM client."""
        loop = self._resource_loop
        if loop is None:
            return
        state = _SHARED_RESOURCES.get(loop)
        self._resource_loop = None
        if state is None:
            return
        state["users"] = max(0, int(state["users"]) - 1)
        if state["users"] == 0:
            client = state.get("client")
            if client is not None:
                await client.aclose()
            _SHARED_RESOURCES.pop(loop, None)

    def _step_image_path(self, step_idx: int) -> str:
        return os.path.join(self.example_result_dir, f"step_{step_idx}.png")

    def _build_messages(
        self,
        *,
        instruction: str,
        actions_history: list[str],
        policy_response: str,
        step_index: int,
        require_guidance: bool = False,
    ) -> List[Dict[str, Any]]:
        step_index = int(step_index)
        rstart_i = max(0, step_index - self.max_reward_image_history_length + 1)

        prev_lines = [f"Step {i + 1}: {actions_history[i]}" for i in range(rstart_i) if i < len(actions_history)]
        previous_actions_str = "\n".join(prev_lines) if prev_lines else "None"

        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "text": f"Previous Actions:\n{previous_actions_str}\n"})

        for i in range(rstart_i, step_index):
            img_b64 = _process_image_bytes(read_file_bytes(self._step_image_path(i)))
            user_content.append({"type": "text", "text": "Image of environment:\n"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            action_text = actions_history[i] if i < len(actions_history) else ""
            user_content.append({"type": "text", "text": f"\nAction of agent:\nStep {i + 1}:\n{action_text}\n"})

        cur_b64 = _process_image_bytes(read_file_bytes(self._step_image_path(step_index)))
        user_content.append({"type": "text", "text": "Agent's current observation:\n"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cur_b64}"}})

        user_content.append({
            "type": "text",
            "text": "\n" + REWARD_INSTRUCTION_TEMPLATE.format(instruction=instruction, response=policy_response)
        })

        nxt_b64 = _process_image_bytes(read_file_bytes(self._step_image_path(step_index + 1)))
        user_content.append({"type": "text", "text": "\nNext observation after executing this action:\n"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{nxt_b64}"}})

        post_instruction = REWARD_INSTRUCTION_POST
        if require_guidance:
            post_instruction += REWARD_GUIDANCE_POST
        user_content.append({"type": "text", "text": post_instruction})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REWARD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages

    async def _call_api(self, messages: List[Dict[str, Any]], vote_id: int) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.api_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text and msg.get("reasoning_content"):
                text = msg["reasoning_content"]
        except Exception as e:
            logger.warning("PRM API call failed (vote=%d): %s", vote_id, e)
            return {
                "score": 0,
                "guidance": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "raw_text": "",
                "ok": False,
            }

        latency_ms = int((time.perf_counter() - start) * 1000)
        score = extract_prm_sign_from_text(text)
        logger.info(
            f"{_ANSI_GREEN}[PRM-API]{_ANSI_RESET} model=%s vote=%d score=%d latency=%dms",
            self.api_model, vote_id, score, latency_ms,
        )
        return {
            "score": score,
            "guidance": extract_prm_guidance_from_text(text),
            "latency_ms": latency_ms,
            "raw_text": text,
            "ok": True,
        }

    async def _vote(self, messages: List[Dict[str, Any]], step_index: int, m: int) -> Dict[str, Any]:
        semaphore = self._get_semaphore()

        async def _single(vote_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._call_api(messages, vote_id)

        votes = await asyncio.gather(*[_single(i) for i in range(max(1, m))])
        scores = [v["score"] for v in votes]
        valid_scores = [int(s) for s in scores if int(s) in (-1, 1)]
        result = {
            "scores": scores,
            "valid_scores": valid_scores,
            "valid_vote_count": len(valid_scores),
            "mean_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0,
            "votes": votes,
        }
        result["guidance"] = select_prm_guidance(votes)
        return result

    async def judge_step(
        self,
        args: Any,
        *,
        instruction: str,
        actions_history: list[str],
        policy_response: str,
        step_index: int,
    ) -> Dict[str, Any]:
        if not getattr(args, "prm_enable", False):
            return {"status": "disabled", "step_index": step_index, "scores": [0], "mean_score": 0.0, "votes": []}

        messages = self._build_messages(
            instruction=instruction,
            actions_history=actions_history,
            policy_response=policy_response,
            step_index=int(step_index),
            require_guidance=bool(getattr(args, "gui_opd_enable", False)),
        )

        m = max(1, int(getattr(args, "prm_m", 1)))
        out = await self._vote(messages, int(step_index), m)
        out["status"] = "ok" if out.get("valid_vote_count", 0) > 0 else "error"
        if getattr(args, "gui_opd_enable", False) and not out.get("guidance"):
            out["status"] = "missing_guidance"
        out["step_index"] = int(step_index)
        if out["status"] != "ok":
            logger.warning("PRM produced no valid votes for step=%d", step_index)

        self.reward_trajectory.append({
            "step_index": int(step_index),
            "prm_scores": out.get("scores", []),
            "prm_mean_score": out.get("mean_score", 0.0),
            "model": self.api_model,
        })
        return out

    def submit_step_judge(
        self,
        args: Any,
        *,
        instruction: str,
        actions_history: list[str],
        policy_response: str,
        step_index: int,
        **_unused: Any,
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.judge_step(
                args=args,
                instruction=instruction,
                actions_history=actions_history,
                policy_response=policy_response,
                step_index=step_index,
            )
        )

    async def collect_step_results(
        self,
        pending_tasks: list[tuple[int, asyncio.Task]],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        if not pending_tasks:
            return [], []
        done = await asyncio.gather(*[task for _, task in pending_tasks], return_exceptions=True)
        step_details: list[dict[str, Any]] = []
        for (step_idx, _), result in zip(pending_tasks, done, strict=False):
            if isinstance(result, Exception):
                step_details.append(
                    {"status": "exception", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []}
                )
                continue
            if not isinstance(result, dict):
                step_details.append(
                    {"status": "invalid_result", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []}
                )
                continue
            result["step_index"] = int(step_idx)
            step_details.append(result)

        step_details.sort(key=lambda x: x.get("step_index", 10**9))
        step_scores = [float(item.get("mean_score", 0.0)) for item in step_details]
        return step_scores, step_details
