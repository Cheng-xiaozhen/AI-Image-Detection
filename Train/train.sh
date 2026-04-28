#!/usr/bin/env bash
set -euo pipefail

# 分布式训练（4xV100）示例
# 如需调整参数，直接修改下方变量或追加参数

DATA_DIR=${DATA_DIR:-"/home/chengxiaozhen/data/Train/BaiXiang"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/chengxiaozhen/Test/SFT-Infra/logs/DINOv3"}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"/home/chengxiaozhen/data/Benchmark /home/chengxiaozhen/data/Eval_Merge"}

MODEL_NAME=${MODEL_NAME:-"dinov3_vith16-vib"}

TRAIN_BS=${TRAIN_BS:-32}
EVAL_BS=${EVAL_BS:-32}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-3} # 全参数微调建议 5e-5，线性学习率调度建议 1e-3
WD=${WD:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
FP16=${FP16:-1}
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
  train.py
  --data_dir "$DATA_DIR"
  --output_dir "${OUTPUT_DIR}/${MODEL_NAME}"
  --model_name "$MODEL_NAME"
  --per_device_train_batch_size "$TRAIN_BS"
  --per_device_eval_batch_size "$EVAL_BS"
  --num_train_epochs "$EPOCHS"
  --learning_rate "$LR"
  --weight_decay "$WD"
  --warmup_ratio "$WARMUP_RATIO"
  --logging_steps "$LOGGING_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --resume
)

if [[ "$FP16" == "1" ]]; then
  CMD+=("--fp16")
fi



if [[ -n "$EVAL_DATA_DIR" ]]; then
  CMD+=("--eval_data_dir" "$EVAL_DATA_DIR")
fi

if [[ -n "$ACCELERATE_CONFIG" ]]; then
  CMD=(accelerate launch --config_file "$ACCELERATE_CONFIG" "${CMD[@]:2}")
fi

printf 'Launching command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
