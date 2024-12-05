import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

# Update font size and other global settings
matplotlib.rcParams.update({'font.size': 20})

def calculate_shared_probability_for_folder(folder_path):
    substring_stats = defaultdict(lambda: defaultdict(int))  # 汇总每个长度的子串出现次数

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue  # 跳过非文件项
        
        with open(file_path, 'rb') as f:
            while True:
                # 每隔 44 字节读取一次
                data = f.read(44)
                if not data:
                    break
                
                # 转换为二进制字符串
                binary_string = ''.join(format(byte, '08b') for byte in data)
                
                # 如果长度不足 100 位，跳过处理
                if len(binary_string) < 100:
                    continue
                
                # 拆分字符串
                mid = len(binary_string) // 2
                first_half = binary_string[:mid]
                second_half = binary_string[mid:][::-1]  # 后半段翻转
                
                # 分别处理两部分
                for part in [first_half, second_half]:
                    if len(part) >= 100:  # 确保有至少 100 位
                        for end in range(13, 101):  # 从第 13 位到第 100 位
                            substring = part[12:end]
                            substring_length = end - 12
                            substring_stats[substring_length][substring] += 1

    # 计算每个长度的共享概率
    shared_probabilities = {}
    for length, substrings in substring_stats.items():
        total_substrings = sum(substrings.values())  # 该长度下所有子串的总出现次数
        shared_probabilities[length] = {
            substring: count / total_substrings
            for substring, count in substrings.items()
        }
    return shared_probabilities

def plot_max_shared_substring(shared_probabilities, save_path):
    max_ratios = {}  # 记录每个子串长度的最大比例

    for length, substrings in shared_probabilities.items():
        max_ratios[length] = max(substrings.values())  # 找到当前长度的最大比例

    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 4))

    lengths = list(max_ratios.keys())
    max_ratios_values = list(max_ratios.values())

    ax.plot(lengths, max_ratios_values, marker='o', label="Max Ratio")
    
    # 设置标签和格式
    ax.set_xlabel("Grouped Exponent Length (bits)", fontsize=18)
    ax.set_ylabel("Shared Bit Ratio", fontsize=18)
    # ax.set_xticks(lengths)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # ax.legend(
    #     frameon=False, 
    #     loc='upper center', 
    #     bbox_to_anchor=(0.5, 1.2), 
    #     ncol=1, 
    #     fontsize=16, 
    #     columnspacing=0.5, 
    #     handletextpad=0.2, 
    #     handlelength=0.8, 
    #     handleheight=0.5, 
    #     borderpad=0.2, 
    #     labelspacing=0.2
    # )

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

shared_probabilities = calculate_shared_probability_for_folder("/home/gaobin/Mahjong-Project/Final-Auto/Chunked-Models/Llama-3.1-8B/bit")

plot_max_shared_substring(shared_probabilities, "/home/gaobin/DNAStorageToolkit/Mahjong/paper-plot/figures/compression_plot.pdf")
