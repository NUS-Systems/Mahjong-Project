import os
import subprocess
import pandas as pd
import torch

# 配置文件路径
DATA_HOME = "/home/gaobin/Mahjong-Project/Final-Auto"
MODEL_NAMES = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Ministral-8B-Instruct-2410", "Qwen2.5-7B-Instruct"]
# VARIATIONS = ["sequential", "bit", "compression"]
# ERROR_PROBS = ["0.01", "0.015", "0.02"]
# REDUNDANCY_VALUES = ["6", "8", "10"]
# COVERAGE = ["5", "10", "15"]
# MODEL_NAMES = ["Llama-3.1-8B"]
VARIATIONS = ["sequential", "bit", "compression"]
ERROR_PROBS = ["0.01", "0.15", "0.02"]
REDUNDANCY_VALUES = ["6", "8", "10", "15", "20", "25", "30", "50"]
COVERAGE = ["5"]

# 函数：加载二进制文件为bfloat16 Tensor
def load_chunk_binary_as_bfloat16(file_path: str) -> torch.Tensor:
    with open(file_path, "rb") as f:
        data = f.read()
    tensor = torch.frombuffer(data, dtype=torch.bfloat16)
    return tensor

def load_recover_binary_as_bfloat16(file_path: str) -> torch.Tensor:
    with open(file_path, "rb") as f:
        data = f.read()
    tensor = torch.frombuffer(data, dtype=torch.bfloat16)
    tensor = torch.nan_to_num(tensor, nan=0.0)
    tensor = torch.clamp(tensor, min=-10, max=10)
    return tensor

# 函数：计算两个Tensor的MSE
def calculate_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个Tensor的形状不匹配，无法计算MSE。")
    mse = torch.mean((tensor1 - tensor2) ** 2)
    return mse.item()

# 函数：计算Hamming距离
def calculate_hamming_distance(file1: str, file2: str) -> int:
    tmp_file1 = f"/tmp/original_{os.path.basename(file1)}.bits"
    tmp_file2 = f"/tmp/recovered_{os.path.basename(file2)}.bits"
    try:
        cmd_xxd_file1 = f"xxd -b {file1} > {tmp_file1}"
        cmd_xxd_file2 = f"xxd -b {file2} > {tmp_file2}"
        os.system(cmd_xxd_file1)
        os.system(cmd_xxd_file2)
        cmd_cmp = f"cmp -l {tmp_file1} {tmp_file2} | wc -l"
        result = subprocess.run(cmd_cmp, shell=True, capture_output=True, text=True)
        distance = int(result.stdout.strip())
    finally:
        if os.path.exists(tmp_file1):
            os.remove(tmp_file1)
        if os.path.exists(tmp_file2):
            os.remove(tmp_file2)
    return distance

# 初始化结果表
results = []

# 遍历所有参数组合
for model_name in MODEL_NAMES:
    for variation in VARIATIONS:
        for error_prob in ERROR_PROBS:
            for redundancy in REDUNDANCY_VALUES:
                for coverage in COVERAGE:
                    chunked_dir = os.path.join(DATA_HOME, "Chunked-Models", model_name, 'sequential')
                    recovered_dir = os.path.join(DATA_HOME, "Final-Recovered-Models", model_name, variation, f"REDUNDANCY_{redundancy}_ERROR_{error_prob}_COVERAGE_{coverage}")
                    
                    if not os.path.exists(chunked_dir) or not os.path.exists(recovered_dir):
                        continue

                    all_hamming_distances = []
                    all_mses = []
                    
                    for chunk_file in os.listdir(chunked_dir):
                        chunk_path = os.path.join(chunked_dir, chunk_file)
                        recovered_path = os.path.join(recovered_dir, chunk_file)
                        
                        if not os.path.exists(chunk_path) or not os.path.exists(recovered_path):
                            continue
                        
                        try:
                            # 计算Hamming距离
                            hamming_distance = calculate_hamming_distance(chunk_path, recovered_path)
                            all_hamming_distances.append(hamming_distance)
                            
                            # 计算MSE
                            tensor1 = load_chunk_binary_as_bfloat16(chunk_path)
                            tensor2 = load_recover_binary_as_bfloat16(recovered_path)
                            mse = calculate_mse(tensor1, tensor2)
                            all_mses.append(mse)
                        except Exception as e:
                            continue
                    
                    # 计算平均值
                    avg_hamming_distance = (sum(all_hamming_distances) / len(all_hamming_distances)) / (5280000/1000*(1+int(redundancy)/100))  if all_hamming_distances else "*"
                    avg_mse = sum(all_mses) / len(all_mses) if all_mses else "*"
                    
                    if variation == 'sequential':
                        baseline = 'Linearized'
                    elif variation == 'bit':
                        baseline = "Bit"
                    elif variation == 'compression':
                        baseline = "Compression"
                    # print(baseline)
                    # 添加结果
                    results.append([
                        model_name, baseline, redundancy, error_prob, coverage, avg_hamming_distance, avg_mse
                    ])

# 保存结果为CSV
output_file = os.path.join(DATA_HOME, "comparison_results.csv")
df = pd.DataFrame(results, columns=["Model", "Scheme", "Redundancy", "Error Rate", "Coverage", "Distance", "MSE"])
df.to_csv(output_file, index=False)

# 打印结果路径
print(f"结果已保存到: {output_file}")
