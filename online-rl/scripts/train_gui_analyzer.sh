#!/bin/bash
# GUI Agent SFT Training Script using ms-swift
#
# Usage:
#   bash scripts/train_gui_analyzer.sh
#
# Requirements:
#   - ms-swift installed: pip install ms-swift
#   - For Qwen3-VL: pip install "transformers>=4.57" "qwen_vl_utils>=0.0.14"
#   - Training data in ms-swift format (use ../data/convert_to_swift_format.py)

set -e

# ==================== Configuration ====================

# Model
MODEL_NAME="path/to/qwen3-vl-8b-thinking"

# Data
TRAIN_DATA="path/to/analyzer-training-data.jsonl"

# Output
OUTPUT_DIR="path/to/gui-analyzer-output"

# ==================== Environment ====================

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NPROC_PER_NODE=16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ==================== Training ====================

mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$MODEL_NAME" \
    --check_model false \
    --dataset "$TRAIN_DATA" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 2 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.0 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --report_to none \
    --deepspeed zero2

echo ""
echo "Training completed! Checkpoints saved to: $OUTPUT_DIR"
