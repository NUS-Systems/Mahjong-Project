#!/bin/sh

# 启用调试模式，输出执行的每一条命令
# set -eux

# 默认值
CODE_DIR=""
DATA_DIR=""
FILE_NAME=""
REDUNDANCY=""
SKIP_RS='0'

# 使用 getopts 解析参数
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--code-dir)
      CODE_DIR="$2"
      shift 2
      ;;
    -d|--data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    -f|--file-name)
      FILE_NAME="$2"
      shift 2
      ;;
    -r|--result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    -p|--redundancy)
      REDUNDANCY="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查必需的参数是否已设置
if [ -z "$CODE_DIR" ] || [ -z "$DATA_DIR" ] || [ -z "$FILE_NAME" ] || [ -z "$RESULT_DIR" ] || [ -z "$REDUNDANCY" ] ; then
  echo "用法: $0 --code-dir <CODE_DIR> --data-dir <DATA_DIR> --file-name <FILE_NAME> --result-dir <RESULT_DIR> --redundancy <REDUNDANCY>"
  exit 1
fi

# 定义目录路径和文件路径
WORK_DIR="/tmp/${FILE_NAME}/REDUNDANCY_${REDUNDANCY}"

CONFIG_FILE="$WORK_DIR/${FILE_NAME}.cfg"

# 创建工作目录
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/input/"
mkdir -p "$WORK_DIR/output/${FILE_NAME}"
mkdir -p "$RESULT_DIR"

# 复制数据文件和代码目录到工作目录
cp "$DATA_DIR/$FILE_NAME" "$WORK_DIR/input/"


# 获取文件大小
file_size=$(stat -c%s "$WORK_DIR/input/$FILE_NAME")
echo "文件大小为: $file_size 字节"

# 设置参数
symbol_size=16
redundancy="$REDUNDANCY"
mapping_scheme=0
input_f="$FILE_NAME"
output_f_encoded="EncodedStrands.txt"
noisy_f="NoisyStrands.txt"
perf_clustered_f="UnderlyingClusters.txt"
clustered_f="ClusteredStrands.txt"
reconstructed_f="ReconstructedStrands.txt"
output_f_decoded="${FILE_NAME}_recovered"
priority_f="NA"

# 创建配置文件
cat <<EOL > "$CONFIG_FILE"
[parameters]
# Parameters for (n, k) Reedsolomon codes. code_length is n, redundancy is n-k
symbol_size      = $symbol_size
redundancy       = $redundancy
file_size        = $file_size
mapping_scheme   = $mapping_scheme

[file_locations] 
input_f	         = $input_f
output_f_encoded = $output_f_encoded
noisy_f  	 = $noisy_f
perf_clustered_f = $perf_clustered_f
clustered_f  	 = $clustered_f
reconstructed_f  = $reconstructed_f
output_f_decoded = $output_f_decoded
priority_f       = $priority_f
EOL


# 1) Encoding data into DNA strands
cp -r "$CODE_DIR/1-encoding-decoding" "$WORK_DIR/"
cd "$WORK_DIR/1-encoding-decoding"
python3 "$WORK_DIR/1-encoding-decoding/codec.py" "$WORK_DIR" "$CONFIG_FILE" 0 "$FILE_NAME" "$SKIP_RS"

cp "$WORK_DIR/output/${FILE_NAME}/EncodedStrands.txt" "$RESULT_DIR/$FILE_NAME"

rm -rf "/tmp/${FILE_NAME}"

# # 2) Simulating wetlab activities which introduce errors
# cp -r "$CODE_DIR/2-simulating_wetlab" "$WORK_DIR/"
# cd "$WORK_DIR/2-simulating_wetlab"
# mkdir -p "$WORK_DIR/output/${FILE_NAME}_P${ERROR_PROB}_N${COVERAGE}"
# python3 "$WORK_DIR/2-simulating_wetlab/naive/noise.py" --N "$COVERAGE" --subs "$ERROR_PROB" --dels "$ERROR_PROB" --inss "$ERROR_PROB" --i "$WORK_DIR/output/${FILE_NAME}/EncodedStrands.txt" --o "$WORK_DIR/output/${FILE_NAME}/UnderlyingClusters.txt"
# python3 "$CODE_DIR/2-simulating_wetlab/naive/shuffle.py" "$WORK_DIR/output/${FILE_NAME}/UnderlyingClusters.txt" "$WORK_DIR/output/${FILE_NAME}/NoisyStrands.txt"


# # 3) Clustering sequenced reads
# cp -r "$CODE_DIR/3-clustering" "$WORK_DIR/"
# cd "$WORK_DIR/3-clustering"
# cp "$WORK_DIR/output/${FILE_NAME}/NoisyStrands.txt" input/.
# cp "$WORK_DIR/output/${FILE_NAME}/UnderlyingClusters.txt" input/.
# make run
# cp "output/ClusteredStrands.txt" "$WORK_DIR/output/${FILE_NAME}/".
# # cd ..

# # 4) Reconstructing original DNA strands from clusters of reads
# cp -r "$CODE_DIR/4-reconstruction" "$WORK_DIR/"
# python3 "$WORK_DIR/4-reconstruction/recon.py" --i "$WORK_DIR/output/${FILE_NAME}/ClusteredStrands.txt" --o "$WORK_DIR/output/${FILE_NAME}/ReconstructedStrands.txt" --coverage "$COVERAGE" --path /dev/shm

# # 5) Decoding data from reconstructed strands
# cd "$WORK_DIR/1-encoding-decoding"
# python3 codec.py "$WORK_DIR" "$CONFIG_FILE" 1 "$FILE_NAME" "$SKIP_RS"

# mkdir -p "$RESULT_DIR/${FILE_NAME}"
# # 6) Copy data to the shared folder
# cp "$WORK_DIR/output/${FILE_NAME}/${FILE_NAME}_recovered" "$RESULT_DIR/${FILE_NAME}"
