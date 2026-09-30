#!/usr/bin/env python3
"""从真实 trajectory.json 重建按图片数量分组的 payload(1图/2图/3图)。

真实训练场景:GUI agent 每步追加一张截图，滑动窗口保留最近 N 张。
  - 轨迹第 0 步 = 真实 1 图(初始截图)
  - 第 1 步     = 真实 2 图
  - 第 2 步起   = 真实 3 图(窗口稳定)
所以按 step 取真实 messages，apply_chat_template 重建，绝不截断造假。

输出:/tmp/payloads_1img.json / _2img.json / _3img_v2.json
格式对齐现有 payloads_3img.json:{text(带占位符), image_data(base64 list), sampling_params, return_logprob}
"""
import json, glob, os, sys
from transformers import AutoProcessor

MODEL = os.environ.get("HF_CKPT")
if not MODEL:
    raise RuntimeError("Set HF_CKPT to the local Hugging Face checkpoint path")
RESULTS_GLOB = "results/slime_gui_8b_partial_async_16gpu_20260611_204755/**/trajectory.json"
TARGET_PER_GROUP = 42          # 与现有 3 图 payload 数量对齐
SAMPLING = {"max_new_tokens": 64, "temperature": 0, "stop_token_ids": [151645]}

def count_imgs(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image")
    return n

def extract_images(messages):
    """按出现顺序抽出所有 image 的 base64 串。"""
    imgs = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and x.get("type") == "image":
                    imgs.append(x["image"])
    return imgs

def main():
    print("加载 processor ...")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    files = glob.glob(RESULTS_GLOB, recursive=True)
    print(f"扫描 {len(files)} 条 trajectory.json")

    groups = {1: [], 2: [], 3: []}
    for f in files:
        if all(len(groups[k]) >= TARGET_PER_GROUP for k in groups):
            break
        try:
            d = json.load(open(f))
        except Exception:
            continue
        traj = d.get("trajectory")
        if not isinstance(traj, list):
            continue
        for step in traj:
            msgs = step.get("messages")
            if not msgs:
                continue
            n = count_imgs(msgs)
            if n in groups and len(groups[n]) < TARGET_PER_GROUP:
                # apply_chat_template 生成带 <|vision_start|><|image_pad|><|vision_end|> 占位符的 text
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                imgs = extract_images(msgs)
                if text.count("<|image_pad|>") != len(imgs):
                    continue  # 占位符与图数不一致，跳过(防御)
                groups[n].append({
                    "text": text,
                    "image_data": imgs,
                    "sampling_params": dict(SAMPLING),
                    "return_logprob": False,
                })

    name = {1: "/tmp/payloads_1img.json", 2: "/tmp/payloads_2img.json", 3: "/tmp/payloads_3img_v2.json"}
    for k in (1, 2, 3):
        out = groups[k]
        json.dump(out, open(name[k], "w"))
        # 报告每组的真实 prompt 规模(用 processor tokenize text 估算)
        if out:
            ntok = len(proc.tokenizer(out[0]["text"]).input_ids)
            print(f"{k}图: {len(out)} 个 payload -> {name[k]}  (样例 text token≈{ntok}, image_data={len(out[0]['image_data'])})")
        else:
            print(f"{k}图: 0 个(未找到，可能轨迹不够)")

if __name__ == "__main__":
    main()
