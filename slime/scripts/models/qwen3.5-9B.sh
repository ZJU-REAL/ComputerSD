MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 12288
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # The 9B checkpoint is dense (no --num-experts), but its HF text_config
   # still declares these Qwen3.5 MoE metadata fields.  slime validates every
   # declared field, so mirror the checkpoint values while leaving MoE layers
   # disabled in get_qwen3_5_spec.
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512

   # Qwen3.5 specific
   --attention-output-gate
)
