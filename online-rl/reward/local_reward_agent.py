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

from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.processing_utils import process_vision_info as slime_process_vision_info


from agents.utils.qwen_vl_utils import smart_resize

logger = logging.getLogger(__name__)

_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"


REWARD_SYSTEM_PROMPT_V1_L2 = """
You are an evaluator for the most recent step of a GUI agent.
You are provided with:
1) the interaction history between the agent and the environment,
2) the agent's objective, and
3) the agent's most recent step to evaluate.
""".strip()


REWARD_INSTRUTION_TEMPLATE = r"""
You are a strict evaluator to evaluate the most recent step of the agent in the following.

Objective of Agent:
{instruction}

Agent's most recent step (reasoning + action):
{response}
"""


REWARD_INSTRUTION_TEMPLATE_POST = r"""
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








def _process_image_bytes_like_policy(image_bytes: bytes) -> str:
    """
    Exactly copy the policy-side image processing:
    - smart_resize with same max_pixels
    - PNG encode
    - base64
    """
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


class Qwen3VLRewardAgent:
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
        self._prm_semaphore: asyncio.Semaphore | None = None

        self.tokenizer = tokenizer
        self.processor = processor

        self._formatter_ready = False

    def _ensure_formatter(self, args: Any) -> None:
        if self._formatter_ready:
            return
        self._formatter_ready = True

        prm_model_path = getattr(args, "prm_model_path", None)
        if not prm_model_path:
            return

        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(prm_model_path, trust_remote_code=True)
        if self.processor is None:
            self.processor = load_processor(prm_model_path, trust_remote_code=True)


    def _step_image_path(self, step_idx: int) -> str:
        return os.path.join(self.example_result_dir, f"step_{step_idx}.png")

    def _get_prm_semaphore(self, args: Any) -> asyncio.Semaphore:
        if self._prm_semaphore is None:
            prm_num_gpus = max(1, int(getattr(args, "prm_num_gpus", 1)))
            prm_num_gpus_per_engine = max(1, int(getattr(args, "prm_num_gpus_per_engine", 1)))
            engine_count = max(1, prm_num_gpus // prm_num_gpus_per_engine)
            base_concurrency = max(1, int(getattr(args, "sglang_server_concurrency", 512)) * engine_count)
            cap = int(getattr(args, "prm_max_concurrency", os.getenv("GUI_PRM_MAX_CONCURRENCY", "32")))
            self._prm_semaphore = asyncio.Semaphore(min(base_concurrency, cap) if cap > 0 else base_concurrency)
        return self._prm_semaphore

    def _build_reward_messages_aligned(
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

        # previous actions text
        prev_lines = [f"Step {i + 1}: {actions_history[i]}" for i in range(rstart_i) if i < len(actions_history)]
        previous_reward_actions_str = "\n".join(prev_lines) if prev_lines else "None"

        user_content: List[Dict[str, Any]] = []
        user_content.append({"type": "text", "text": f"Previous Actions:\n{previous_reward_actions_str}\n"})

        # include (optional) last history steps images, then current obs image, then next obs image
        for i in range(rstart_i, step_index):
            img_b64 = _process_image_bytes_like_policy(read_file_bytes(self._step_image_path(i)))
            user_content.append({"type": "text", "text": "Image of environment:\n"})
            user_content.append({"type": "image", "image": f"data:image/png;base64,{img_b64}"})
            action_text = actions_history[i] if i < len(actions_history) else ""
            user_content.append({"type": "text", "text": f"\nAction of agent:\nStep {i + 1}:\n{action_text}\n"})

        # current observation
        cur_b64 = _process_image_bytes_like_policy(read_file_bytes(self._step_image_path(step_index)))
        user_content.append({"type": "text", "text": "Agent's current observation:\n"})
        user_content.append({"type": "image", "image": f"data:image/png;base64,{cur_b64}"})

        # instruction + response
        user_content.append({
            "type": "text",
            "text": "\n" + REWARD_INSTRUTION_TEMPLATE.format(instruction=instruction, response=policy_response)
        })

        # next observation after executing this action
        nxt_b64 = _process_image_bytes_like_policy(read_file_bytes(self._step_image_path(step_index + 1)))
        user_content.append({"type": "text", "text": "\nNext observation after executing this action:\n"})
        user_content.append({"type": "image", "image": f"data:image/png;base64,{nxt_b64}"})

        post_instruction = REWARD_INSTRUTION_TEMPLATE_POST
        if require_guidance:
            post_instruction += REWARD_GUIDANCE_POST
        user_content.append({"type": "text", "text": post_instruction})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": REWARD_SYSTEM_PROMPT_V1_L2}]},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _build_payload_like_policy(
        self,
        *,
        args: Any,
        messages: List[Dict[str, Any]],
        vote_id: int,
    ) -> Tuple[Dict[str, Any], str, int, int]:
        if self.tokenizer is None:
            raise RuntimeError("PRM tokenizer is not initialized.")

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = {
            "temperature": float(getattr(args, "prm_temperature", 0.0) or 0.0),
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 256) or 256),
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
            "sampling_seed": int(getattr(args, "rollout_seed", 42)) * 1000 + int(vote_id),
        }

        payload: Dict[str, Any] = {"sampling_params": sampling_params, "return_logprob": True}
        image_count = 0
        if self.processor is not None:
            multimodal_inputs = slime_process_vision_info(messages, self.processor) or {}
            images = multimodal_inputs.get("images") or []
            image_count = len(images)
            if images:
                payload["image_data"] = [encode_image_for_rollout_engine(img) for img in images]

        input_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if payload.get("image_data"):
            payload["text"] = prompt_text
        else:
            payload["input_ids"] = input_ids
        return payload, prompt_text, len(input_ids), image_count

    def _decode_text(self, output: Any) -> str:
        """
        Decode exactly like policy: output_token_logprobs -> item[1] FIRST,
        fall back to output["text"] only if logprobs are absent.
        """
        if not isinstance(output, dict):
            return "" if output is None else str(output)

        # Priority 1: decode from output_token_logprobs (same as policy)
        meta = output.get("meta_info", {})
        if isinstance(meta, dict) and self.tokenizer is not None:
            otlp = meta.get("output_token_logprobs")
            if isinstance(otlp, list) and otlp:
                output_tokens = [item[1] for item in otlp]
                return self.tokenizer.decode(output_tokens)

        # Priority 2: fall back to text
        text = output.get("text")
        if isinstance(text, str):
            return text

        return ""

    async def _query_prm_once(
        self,
        args: Any,
        messages_for_prm: List[Dict[str, Any]],
        step_index: int,
        vote_id: int,
    ) -> Dict[str, Any]:
        prm_router_ip = getattr(args, "prm_router_ip", None)
        prm_router_port = getattr(args, "prm_router_port", None)
        if not prm_router_ip or not prm_router_port:
            return {"score": 0, "guidance": "", "latency_ms": 0, "raw_text": "", "ok": False}

        payload, prompt_text, in_len, img_cnt = await asyncio.to_thread(
            self._build_payload_like_policy,
            args=args,
            messages=messages_for_prm,
            vote_id=vote_id,
        )

        start = time.perf_counter()
        try:
            output = await post(f"http://{prm_router_ip}:{prm_router_port}/generate", payload, max_retries=30)
        except Exception:
            return {
                "score": 0,
                "guidance": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "raw_text": "",
                "ok": False,
            }

        text = self._decode_text(output)
        logger.info(f"{_ANSI_RED}[PRM]{_ANSI_RESET} step=%s vote=%s output:\n%r", int(step_index), int(vote_id), text)

        return {
            "score": extract_prm_sign_from_text(text),
            "guidance": extract_prm_guidance_from_text(text),
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "raw_text": text,
            "ok": True,
        }

    async def _prm_vote(
        self,
        args: Any,
        messages_for_prm: List[Dict[str, Any]],
        step_index: int,
        m: int,
    ) -> Dict[str, Any]:
        semaphore = self._get_prm_semaphore(args)

        async def _single(vote_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._query_prm_once(args, messages_for_prm, step_index, vote_id)

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
        if not getattr(args, "prm_router_ip", None) or not getattr(args, "prm_router_port", None):
            return {"status": "disabled_no_router", "step_index": step_index, "scores": [0], "mean_score": 0.0, "votes": []}

        self._ensure_formatter(args)

        reward_messages = self._build_reward_messages_aligned(
            instruction=instruction,
            actions_history=actions_history,
            policy_response=policy_response,
            step_index=int(step_index),
            require_guidance=bool(getattr(args, "gui_opd_enable", False)),
        )

        out = await self._prm_vote(
            args=args,
            messages_for_prm=reward_messages,
            step_index=int(step_index),
            m=max(1, int(getattr(args, "prm_m", 3))),
        )

        out["status"] = "ok" if out.get("valid_vote_count", 0) > 0 else "error"
        if getattr(args, "gui_opd_enable", False) and not out.get("guidance"):
            out["status"] = "missing_guidance"
        out["step_index"] = int(step_index)

        self.reward_trajectory.append(
            {
                "step_index": int(step_index),
                "reward_messages": copy.deepcopy(reward_messages),
                "prm_scores": out.get("scores", []),
                "prm_mean_score": out.get("mean_score", 0.0),
            }
        )
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
