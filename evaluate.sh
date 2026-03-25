#!/usr/bin/env bash
set -euo pipefail

# 分布式评估示例
# 如需调整参数，直接修改下方变量或追加参数

CHECKPOINT=${CHECKPOINT:-"/home/chengxiaozhen/Test/SFT-Infra/logs/convnext2_tiny/final_model"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/chengxiaozhen/Test/SFT-Infra/logs/convnext2_tiny/eval"}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"/home/chengxiaozhen/data/Benchmark/Chameleon"}

MODEL_NAME=${MODEL_NAME:-"convnext2_tiny"}
NUM_CLASSES=${NUM_CLASSES:-1}

EVAL_BS=${EVAL_BS:-32}
LOGGING_STEPS=${LOGGING_STEPS:-50}
FP16=${FP16:-0} # 是否使用半精度评估，1表示启用，0表示禁用
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-""}
MIXED_PRECISION="no"

if [[ "$FP16" == "1" ]]; then
	MIXED_PRECISION="fp16"
fi

CMD=(
	accelerate launch
	--num_processes 4
	--mixed_precision "$MIXED_PRECISION"
	evaluate.py
	--checkpoint "$CHECKPOINT"
	--output_dir "$OUTPUT_DIR"
	--eval_data_dir "$EVAL_DATA_DIR"
	--model_name "$MODEL_NAME"
	--num_classes "$NUM_CLASSES"
	--per_device_eval_batch_size "$EVAL_BS"
	--logging_steps "$LOGGING_STEPS"
	--dataloader_num_workers "$DATALOADER_NUM_WORKERS"
)

if [[ "$FP16" == "1" ]]; then
	CMD+=("--fp16")
fi

if [[ -n "$ACCELERATE_CONFIG" ]]; then
	CMD=(accelerate launch --config_file "$ACCELERATE_CONFIG" "${CMD[@]:2}")
fi

printf 'Launching command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
