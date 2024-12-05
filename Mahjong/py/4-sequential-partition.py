import os
from pathlib import Path
import argparse
import torch
import struct
from safetensors import safe_open
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import sys
import itertools

def extract_payload(file_path):
    """提取除去metadata的payload部分"""
    with open(file_path, 'rb') as f:
        # 读取metadata长度（假设前8字节表示metadata的长度）
        metadata_length_bytes = f.read(8)
        metadata_length = int.from_bytes(metadata_length_bytes, 'little')
        print(f.tell())
        # 跳过metadata部分
        f.seek(metadata_length)
        print(f.tell())
        # 读取payload部分
        payload = f.read()
    
    return payload

def split_payload(payload, output_dir, base_file_name, offcut_dir, chunk_size=2 * 1024 * 1024):
    """将提取的payload部分分成多个chunk并保存。如果剩余数据小于chunk size，则存储到offcut dir中，但保留chunk_number。"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(offcut_dir, exist_ok=True)
    
    chunk_number = 0
    start = 0
    
    while start < len(payload):
        # 计算剩余数据的大小
        remaining_data_size = len(payload) - start
        
        # 如果剩余数据小于chunk_size，则保存到offcut_dir
        if remaining_data_size < chunk_size:
            chunk_data = payload[start:]
            offcut_file_path = os.path.join(offcut_dir, f"{base_file_name}.part.{chunk_number}")
            with open(offcut_file_path, 'wb') as offcut_file:
                offcut_file.write(chunk_data)
            print(f"剩余数据小于chunk_size，保存到: {offcut_file_path}")
            break
        
        # 正常情况，按chunk_size分块
        end = start + int(chunk_size)
        chunk_data = payload[start:end]
        
        # 保存chunk文件
        chunk_file_path = os.path.join(output_dir, f"{base_file_name}.part.{chunk_number}")
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk_data)
        
        chunk_number += 1
        start = end

def process_and_split_file(input_file, output_dir, offcut_dir, chunk_size=2 * 1024 * 1024):
    """处理文件，提取payload并进行分块"""
    # 提取文件名作为base_file_name
    base_file_name = os.path.basename(input_file)
    
    # 提取payload
    payload = extract_payload(input_file)
    
    # 对payload进行分块并保存
    split_payload(payload, output_dir, base_file_name, offcut_dir, chunk_size)


def main():
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="Extract payload from a file and split it into smaller chunks.")
    
    # 添加命令行参数
    parser.add_argument('input_file', help='The absolute path of the input file to be processed.')
    parser.add_argument('output_dir', help='The absolute path of the directory to output the chunks.')
    parser.add_argument('--chunk_size', type=float, default=2 * 1024 * 1024, help='Size of each chunk in bytes (default: 2 MB).')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of files to process in each batch (default: 10).')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input_file):
        print("输入的文件不存在，请检查路径！")
        return
    
    number_on_strand = 22
    root = args.output_dir

    sequential_dir = os.path.join(root, "sequential")
    component_interleave_dir = os.path.join(root, "component")
    binary_interleave_dir = os.path.join(root, "bit")
    binary_compress_dir = os.path.join(root, "compression")
    offcut_dir = os.path.join(root, "offcut")

    # 运行文件处理和分割
    process_and_split_file(args.input_file, sequential_dir, offcut_dir, args.chunk_size)

if __name__ == "__main__":
    main()