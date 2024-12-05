import os
import shutil
import subprocess
from pathlib import Path

# 配置文件路径
DATA_HOME = "/home/gaobin/Mahjong-Project/Final-Auto"
MODEL_NAMES = ["Llama-3.1-8B"]
VARIATIONS = ["sequential", "bit", "compression"]
ERROR_PROBS = ["0.01", "0.015", "0.02"]
REDUNDANCY_VALUES = ["6", "8", "10", "15", "20", "25", "30", "50"]
COVERAGE = ["5", "10", "15"]
CODE_DIR = os.path.join(DATA_HOME, "DNAStorageToolkit")

# 定义输出目录
FINAL_RECOVERED_DIR = os.path.join(DATA_HOME, "Final-Recovered-Models")

# 确保输出目录存在
os.makedirs(FINAL_RECOVERED_DIR, exist_ok=True)

# 遍历所有参数组合
for model_name in MODEL_NAMES:
    for variation in VARIATIONS:
        for error_prob in ERROR_PROBS:
            for redundancy in REDUNDANCY_VALUES:
                for coverage in COVERAGE:
                    # 定义输入文件夹路径
                    data_dir = os.path.join(DATA_HOME, "Chunked-Models", model_name, variation)
                    bit_dir = os.path.join(DATA_HOME, "Chunked-Models", model_name, 'bit')
                    encoded_dir = os.path.join(DATA_HOME, "Encoded-Models", model_name, variation, f"REDUNDANCY_{redundancy}")
                    recovered_dir = os.path.join(DATA_HOME, "Recovered-Models", model_name, variation,
                                                f"REDUNDANCY_{redundancy}_ERROR_{error_prob}_COVERAGE_{coverage}")
                    
                    # 定义输出路径
                    output_dir = os.path.join(FINAL_RECOVERED_DIR, model_name, variation, 
                                            f"REDUNDANCY_{redundancy}_ERROR_{error_prob}_COVERAGE_{coverage}")
                    os.makedirs(output_dir, exist_ok=True)

                    if variation == "sequential":
                        # 直接拷贝 sequential 文件夹内容
                        print(f"Copying sequential results for {model_name}, redundancy {redundancy}, error {error_prob} coverage {coverage}")
                        if os.path.exists(recovered_dir):
                            for file_name in os.listdir(recovered_dir):
                                src_file = os.path.join(recovered_dir, file_name)
                                dst_file = os.path.join(output_dir, file_name)
                                shutil.copy2(src_file, dst_file)
                        else:
                            print(f"Warning: Sequential source directory {recovered_dir} does not exist.")
                    
                    elif variation == "bit":
                        # 调用 bit-to-sequential.py
                        print(f"Processing bit variation for {model_name}, redundancy {redundancy}, error {error_prob} coverage {coverage}")
                        subprocess.run([
                            "python3", os.path.join(CODE_DIR, "bit-to-sequential.py"),
                            recovered_dir, output_dir,
                            "--segment_size", "22"
                        ], check=True)
                    
                    elif variation == "compression":
                        # 调用 decompress_bit_interleave.py 处理压缩文件夹
                        interleave_output_dir =  os.path.join(DATA_HOME, "Intermediate-Bit", model_name, variation,
                                                f"REDUNDANCY_{redundancy}_ERROR_{error_prob}_COVERAGE_{coverage}")
                        os.makedirs(interleave_output_dir, exist_ok=True)
                        
                        print(f"Decompressing and processing compression variation for {model_name}, redundancy {redundancy}, error {error_prob}")
                        subprocess.run([
                            "python3", os.path.join(CODE_DIR, "decompress_bit_interleave.py"),
                            bit_dir, recovered_dir, interleave_output_dir
                        ], check=True)
                        
                        # 调用 bit-to-sequential.py
                        subprocess.run([
                            "python3", os.path.join(CODE_DIR, "bit-to-sequential.py"),
                            interleave_output_dir, output_dir,
                        ], check=True)

print("Processing complete. All results saved to Final-Recovered-Models.")
