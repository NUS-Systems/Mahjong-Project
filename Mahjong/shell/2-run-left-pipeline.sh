#!/bin/sh
set -eux

# Parse command line arguments
EXAMPLE_MODE=false
EXAMPLE_FILE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --example)
      EXAMPLE_MODE=true
      shift
      ;;
    --file)
      EXAMPLE_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Define the parameters
# MODEL_NAMES=("Llama-3.2-3B" "Ministral-8B-Instruct-2410" "Qwen2.5-7B-Instruct")
MODEL_NAMES=("Llama-3.2-1B")
VARIATIONS=("sequential" "bit" "compression")  # 变化类型
# VARIATIONS=("sequential")  # 变化类型
ERROR_PROBS=("0.01" "0.02")
REDUNDANCY_VALUES=("10")  # 多个冗余系数
COVERAGE="5"
CODE_DIR='/mydata/DNAStorageToolkit'

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for VARIATION in "${VARIATIONS[@]}"; do
    for ERROR_PROB in "${ERROR_PROBS[@]}"; do
      for REDUNDANCY in "${REDUNDANCY_VALUES[@]}"; do
        DATA_DIR="/mydata/Chunked-Models/$MODEL_NAME/$VARIATION"
        ENCODED_DIR="/mydata/Encoded-Models/$MODEL_NAME/$VARIATION/REDUNDANCY_${REDUNDANCY}"
        RECOVERED_DIR="/mydata/Recovered-Models/$MODEL_NAME/$VARIATION/REDUNDANCY_${REDUNDANCY}_ERROR_${ERROR_PROB}_COVERAGE_${COVERAGE}"

        # rm -rf "$RECOVERED_DIR"
        mkdir -p "$RECOVERED_DIR"

        export CODE_DIR ENCODED_DIR RECOVERED_DIR REDUNDANCY ERROR_PROB COVERAGE

        # 找出 $ENCODED_DIR 中存在但 $RECOVERED_DIR 中不存在的文件
        diff_files=$(comm -23 \
          <(find "$ENCODED_DIR" -type f -exec basename {} \; | sort) \
          <(find "$RECOVERED_DIR" -type f -exec basename {} \; | sort))

        # 如果启用了 example 模式，并且指定了文件名
        if [ "$EXAMPLE_MODE" = true ]; then
          if [ -n "$EXAMPLE_FILE" ]; then
            # 检查 EXAMPLE_FILE 是否在 diff_files 中
            if echo "$diff_files" | grep -q "$EXAMPLE_FILE"; then
              diff_files="$EXAMPLE_FILE"
            else
              echo "指定的文件 $EXAMPLE_FILE 不在待处理文件列表中。"
              continue
            fi
          else
            # 如果没有指定文件名，默认取第一个文件
            diff_files=$(echo "$diff_files" | head -n 1)
          fi
        fi

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
  done
done

echo "All tasks for all models and error rates have been completed."
