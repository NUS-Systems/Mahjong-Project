import os
import re

def find_missing_files(directory, ranges):
    # 存储所有匹配的文件
    files_found = set()

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(directory):
        # 确保文件名符合格式
            # 提取第一个数值：位于第一个 `-` 和第二个 `-` 之间
        first_dash = file_name.find('-')  # 第一个 '-'
        second_dash = file_name.find('-', first_dash + 1)  # 第二个 '-'
        num1_str = file_name[first_dash + 1:second_dash]  # 提取第一个数字

          # 提取第二个数值：位于 `.part.` 后
        part_index = file_name.index('.part.')  # 找到 '.part.'
        num2_str = file_name[part_index + 6:]  # 提取第二个数字

            # 转换为整数
        num1 = int(num1_str)
        num2 = int(num2_str)

            # 保存匹配到的文件
        files_found.add((num1, num2))
        print(f'{num1}, {num2}')

    # 统计文件总数和缺失的文件
    total_expected_files = 0   # 预计应该存在的文件总数
    total_matched_files = len(files_found)  # 已匹配到的文件总数
    missing_files = []         # 存放未匹配到的文件名

    # 遍历每个范围，找出缺失的文件
    for num1, (start2, end2) in ranges.items():
        for num2 in range(start2, end2 + 1):
            total_expected_files += 1
            if (num1, num2) not in files_found:
                # 不使用前导零，直接将数字插入字符串
                missing_files.append(f'model-0000{num1}-of-00004.safetensors.part.{num2}')

    total_missing_files = total_expected_files - total_matched_files  # 缺失的文件数

    return missing_files, total_matched_files, total_missing_files
# 使用示例
directory = '/mydata/Collected-Models/Llama-3.1-8B/REDUNDANCY_15_ERROR_0.02_COVERAGE_10/'  # 替换为你文件夹的路径

# 定义每个第一个数值对应的第二个数值范围
ranges = {
    1: (0, 3164),  # 比如第一个数值为 1 时，第二个数值的范围是 0 到 50
    2: (0, 3178), # 比如第一个数值为 2 时，第二个数值的范围是 0 到 100
    3: (0, 3125),  # 比如第一个数值为 3 时，第二个数值的范围是 0 到 30
    4: (0, 742)   # 比如第一个数值为 4 时，第二个数值的范围是 0 到 10
}

missing_files, total_matched_files, total_missing_files = find_missing_files(directory, ranges)

if missing_files:
    print("缺失的文件有：")
    for file in missing_files:
        print(file)
else:
    print("没有发现缺失的文件。")

# 输出匹配到的文件总数和缺失的文件数
print(f"匹配到的文件总数: {total_matched_files}")
print(f"缺失的文件总数: {total_missing_files}")

def count_files_in_directory(directory):
    # 列出目录中的所有文件和文件夹
    files_and_dirs = os.listdir(directory)

    # 过滤掉目录中的子文件夹，只保留文件
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]

    # 返回文件数量
    return len(files)

total_files = count_files_in_directory(directory)

print(f"该文件夹下的文件总数是: {total_files}")