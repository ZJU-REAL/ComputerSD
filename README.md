# ComputerSD

ComputerSD 的 GUI agent 在线训练代码。`online-rl/` 包含 agent、环境客户端、rollout、reward 与训练入口；`slime/` 和 `Megatron-LM/` 是训练依赖源码。

## 目录

- `online-rl/scripts/`：训练与辅助启动脚本、YAML 配置。
- `online-rl/train_fully_async.py`：异步训练入口。
- `online-rl/train_serial_opd.py`：串行 OPD 训练入口。
- `online-rl/evaluation_examples/`：GUI 任务元数据。
- `slime/`、`Megatron-LM/`：底层训练框架。

## 运行前配置

脚本面向 Linux、多 GPU、Ray、SGLang、Megatron-LM 与可访问的 GUI 环境服务。各节点需能读取相同的模型和配置路径。先安装当前仓库所需依赖，再设置实际路径与服务地址，例如：

```bash
export HF_CKPT=path/to/qwen3-vl-8b-thinking
export ANALYZER_MODEL_PATH=path/to/gui-analyzer
export GUI_ENV_SERVER_URL=http://gui-env-host/osworld-node
```

`path/to/...` 均为占位路径，运行前应替换为实际路径；不需要 analyzer 的训练可只设置 `HF_CKPT`。输出目录、GPU 数量和并发度可通过脚本中的同名环境变量覆盖。

## 主要训练脚本

从仓库根目录运行：

```bash
# GRPO；ENABLE_PRM=0 可关闭 analyzer 奖励
bash online-rl/scripts/gui_qwen3vl_16gpu_async_grpo.sh

# GRPO + online self-distillation
bash online-rl/scripts/gui_qwen3vl_16gpu_async_grpo_opd.sh

# GiGPO
bash online-rl/scripts/gui_qwen3vl_16gpu_async_gigpo.sh

# EvoCUA 对应配置
bash online-rl/scripts/gui_evocua_8b_16gpu_async_grpo_opd.sh
```

这些脚本共用 `online-rl/scripts/gui_qwen3vl_16gpu_async_common.sh`。默认启动本地 Ray head；已有 Ray 集群时，可设置 `RESET_LOCAL_RAY=1` 先停止本地 Ray。请在专用训练节点上运行。

`online-rl/scripts/gui_qwen3vl_8b_eval_only.sh`、`gui_qwen3vl_8b_sample_only.sh` 与 EvoCUA 评测封装仍引用未包含在当前代码快照中的 `gui_qwen3vl_8b_fast.sh`，因此暂不能直接执行。发布评测流程前需要补齐该入口。
