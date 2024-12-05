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

MODEL_NAMES=("Llama-3.1-8B")  # 多个模型名称
VARIATIONS=("bit" "sequential")  # 变化类型
REDUNDANCY_VALUES=("10")  # 多个冗余系数
CODE_DIR='/mydata/DNAStorageToolkit'

# Loop through each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  # Loop through each variation
  for VARIATION in "${VARIATIONS[@]}"; do
    # Loop through each redundancy value
    for REDUNDANCY in "${REDUNDANCY_VALUES[@]}"; do
      DATA_DIR="/mydata/Chunked-Models/$MODEL_NAME/$VARIATION"  # 动态选择文件夹
      RESULT_DIR="/mydata/Encoded-Models/$MODEL_NAME/$VARIATION/REDUNDANCY_${REDUNDANCY}"

    #   rm -rf "$RESULT_DIR"
      mkdir -p "$RESULT_DIR"

      export CODE_DIR DATA_DIR RESULT_DIR REDUNDANCY

      diff_files=$(comm -23 \
          <(find "$DATA_DIR" -type f -exec basename {} \; | sort) \
          <(find "$RESULT_DIR" -type f -exec basename {} \; | sort))


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
        
      if [ -n "$diff_files" ]; then
        # Use GNU Parallel to execute the encoding pipeline in parallel
        echo "$diff_files" | parallel bash 1-encoding-pipeline.sh --code-dir "$CODE_DIR" --data-dir "$DATA_DIR" --file-name {} --result-dir "$RESULT_DIR" --redundancy "$REDUNDANCY"
      else
        echo "没有需要处理的文件。"
      fi
      # Print completion message for the current variation and redundancy
      echo "All tasks for $MODEL_NAME - $VARIATION with REDUNDANCY $REDUNDANCY have been completed."
    done
  done
done

# Print overall completion message
echo "All tasks for all models, variations, and redundancy values have been completed."
