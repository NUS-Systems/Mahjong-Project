#!/bin/sh
set -eux

# 定义参数
MODEL_NAMES=("Llama-3.2-1B")
ERROR_PROBS=("0.01")
REDUNDANCY="10"
COVERAGE="10"
CODE_DIR='/mydata/DNAStorageToolkit'

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for ERROR_PROB in "${ERROR_PROBS[@]}"; do

    DATA_DIR="/mydata/Chunked-Models/$MODEL_NAME/sequential"
    ENCODED_DIR="/mydata/Encoded-Models/$MODEL_NAME/sequential/REDUNDANCY_${REDUNDANCY}"
    RECOVERED_DIR="/mydata/Recovered-Models/$MODEL_NAME/sequential/REDUNDANCY_${REDUNDANCY}_ERROR_${ERROR_PROB}_COVERAGE_${COVERAGE}"

    export CODE_DIR ENCODED_DIR RECOVERED_DIR REDUNDANCY ERROR_PROB COVERAGE

    # 找出 $ENCODED_DIR 中存在但 $RECOVERED_DIR 中不存在的文件
    diff_files=$(comm -23 \
      <(find "$ENCODED_DIR" -type f -exec basename {} \; | sort) \
      <(find "$RECOVERED_DIR" -type f -exec basename {} \; | sort))

    # 如果有需要处理的文件，使用 parallel 执行 pipeline
    if [ -n "$diff_files" ]; then
      echo "$diff_files" | parallel bash 2-left-pipeline.sh --code-dir "$CODE_DIR" --data-dir "$DATA_DIR" --encoded-dir "$ENCODED_DIR" --file-name {} --recovered-dir "$RECOVERED_DIR" --redundancy "$REDUNDANCY" --error-prob "$ERROR_PROB" --coverage "$COVERAGE"
    else
      echo "没有需要处理的文件。"
    fi

    # 打印完成消息
    echo "模型 $MODEL_NAME 和错误率 $ERROR_PROB 的任务已完成。"

  done
done

echo "所有模型和错误率的任务都已完成。"
