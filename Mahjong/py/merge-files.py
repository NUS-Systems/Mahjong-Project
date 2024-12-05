#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def merge_files_with_metadata(metadata_file: str, sorted_files: list, output_file: str):
    # 确保输出文件夹存在
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

    with open(metadata_file, 'rb') as metafile, open(output_file, 'wb') as outfile:
        # 读取metadata长度（假设前8字节为metadata长度）
        metadata_length_bytes = metafile.read(8)
        metadata_length = int.from_bytes(metadata_length_bytes, 'little')

        # 写入metadata部分到输出文件
        metafile.seek(0)  # 重置读取位置到文件开头
        metadata = metafile.read(metadata_length)
        outfile.write(metadata)

        # 按顺序合并所有文件内容
        for file in sorted_files:
            with open(file, 'rb') as infile:
                outfile.write(infile.read())

    print(f"处理完成，生成文件：{output_file}")

def process_folders(part_folder: str, additional_folder: str, metadata_folder: str, output_folder: str):
    # 收集所有文件，按基名分组
    file_groups = {}

    # 处理part_folder中的文件
    for file in os.listdir(part_folder):
        if ".part." in file:
            base_name, _, part_number = file.rpartition(".part.")
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(os.path.join(part_folder, file))

    # 处理additional_folder中的文件
    for file in os.listdir(additional_folder):
        if ".part." in file:
            base_name, _, part_number = file.rpartition(".part.")
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(os.path.join(additional_folder, file))

    # 遍历分组处理
    for base_name, file_list in file_groups.items():
        # 查找对应的metadata文件
        metadata_file = os.path.join(metadata_folder, base_name)
        if not os.path.isfile(metadata_file):
            print(f"警告：找不到基名为 {base_name} 的metadata文件，跳过该组文件。")
            continue

        # 输出文件路径
        output_file = os.path.join(output_folder, base_name)

        # 按part编号排序
        sorted_files = sorted(file_list, key=lambda x: int(x.rpartition(".part.")[2]))

        # 合并文件并写入输出
        merge_files_with_metadata(metadata_file, sorted_files, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_with_metadata.py <part_folder> <additional_folder> <metadata_folder> <output_folder>")
        sys.exit(1)

    part_folder = sys.argv[1]
    additional_folder = sys.argv[2]
    metadata_folder = sys.argv[3]
    output_folder = sys.argv[4]

    if not os.path.isdir(part_folder):
        print(f"错误：待合并文件夹 '{part_folder}' 不存在。")
        sys.exit(1)

    if not os.path.isdir(additional_folder):
        print(f"错误：额外文件夹 '{additional_folder}' 不存在。")
        sys.exit(1)

    if not os.path.isdir(metadata_folder):
        print(f"错误：metadata文件夹 '{metadata_folder}' 不存在。")
        sys.exit(1)

    process_folders(part_folder, additional_folder, metadata_folder, output_folder)
