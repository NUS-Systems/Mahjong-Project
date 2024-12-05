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
import ctypes

def load_bfloat16_tensor(file_path, shape=None):
    """
    从文件中读取 bfloat16 数据，并将其转换为 PyTorch 的 bfloat16 tensor。
    
    参数:
    - file_path: 包含 bfloat16 数据的文件路径
    - shape: (可选) 如果你知道数据的形状，可以传入一个 tuple 进行 reshape
    
    返回:
    - PyTorch 的 bfloat16 tensor
    """
    # Step 1: 读取文件中的二进制数据
    with open(file_path, 'rb') as f:
        byte_data = f.read()

    # Step 2: 确保文件大小是2的倍数，因为每个bfloat16占用2个字节
    if len(byte_data) % 2 != 0:
        raise ValueError("文件大小不是2的倍数，可能包含无效的 bfloat16 数据")

    # Step 3: 将二进制数据转换为 numpy 的 uint16 数组
    # np_data = np.frombuffer(file_path, dtype=np.uint16)
    
    # Step 4: 将 numpy 数组转换为 PyTorch 的 bfloat16 张量
    tensor = torch.frombuffer(byte_data, dtype=torch.bfloat16)

    # Step 5: 如果提供了 shape，重新调整张量的形状
    if shape:
        tensor = tensor.reshape(shape)

    return tensor

# 将位列表打包为字节列表，按 8 位为 1 字节
def pack_bits_to_bytes(bits):
    byte_list = []
    current_byte = 0
    bit_count = 0

    for bit in bits:
        current_byte = (current_byte << 1) | bit  # 将当前位移入当前字节
        bit_count += 1

        # 每满 8 位，打包成一个字节
        if bit_count == 8:
            byte_list.append(current_byte)
            current_byte = 0
            bit_count = 0

    # 如果最后不足 8 位，进行填充
    if bit_count > 0:
        current_byte = current_byte << (8 - bit_count)  # 左移并填充 0
        byte_list.append(current_byte)

    return byte_list

# 方案1 - 按顺序写入 bfloat16 的二进制文件
def write_bfloat16_binary_sequential(tensor, file_path):
    # 获取数据指针
    data_ptr = tensor.data_ptr()

    # 计算总字节数
    total_bytes = tensor.numel() * tensor.element_size()
    
    # 转换数据指针为 ctypes 指针
    c_buffer = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte * total_bytes))

    # 将数据写入文件
    with open(file_path, "wb") as f:
        f.write(bytearray(c_buffer.contents))

    print(f"BFloat16 Tensor buffer 已成功写入文件：{file_path}")

# 读取 bfloat16 二进制文件并恢复为 tensor
def read_bfloat16_binary_sequential(file_path, tensor_size):
    recovered_tensor = []
    with open(file_path, 'rb') as f:
        for _ in range(tensor_size):
            binary_value = f.read(2)  # 读取 2 个字节
            int_value = struct.unpack('>H', binary_value)[0]  # 解析成无符号 16 位整数
            # 将无符号整数转换回 bfloat16
            float_value = torch.tensor([int_value], dtype=torch.uint16).view(torch.bfloat16).item()
            recovered_tensor.append(float_value)
    
    return torch.tensor(recovered_tensor, dtype=torch.bfloat16)

def unpack_bytes_to_bits(byte_list):
    # 将字节列表转换为位列表
    bits = []
    for byte in byte_list:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 0x1)  # 从高位到低位提取每一位
    return bits

def read_bfloat16_binary_interleave_bits(file_path, output_path, segment_size=4):
    def pack_bits_to_bytes(bits):
        """将一系列二进制位转换为字节序列"""
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= (bits[i + j] << (7 - j))
            byte_array.append(byte)
        return byte_array
    recovered_tensor = []  # 存储恢复的bfloat16值
    payload = b''
    with open(file_path, 'rb') as f:
        payload = f.read()
    
    segment_size = segment_size*2
    with open(output_path, 'wb') as f:
        for i in range(0, len(payload), segment_size):
            half_size = segment_size // 2

            interleaved_bytes = payload[i:i + segment_size]
            
            interleaved_bits = unpack_bytes_to_bits(interleaved_bytes)

            # 分割位数组为前半段和后半段
            first_half_bits = interleaved_bits[:11 * 16]
            second_half_bits = interleaved_bits[11 * 16:]

            # 对第二半段的位进行反转（恢复原始顺序）
            second_half_bits.reverse()

            # 从前半段恢复符号位、指数位和尾数位
            sign_bits_first_half = first_half_bits[:11]
            exponent_bits_first_half = [[] for _ in range(11)]
            fraction_bits_first_half = [[] for _ in range(11)]

            index = 11

            # 恢复第一半部分的指数位
            for j in range(8):
                for k in range(11):
                    exponent_bits_first_half[k].append(first_half_bits[index])
                    index += 1

            # 恢复第一半部分的尾数位
            for j in range(7):
                for k in range(11):
                    fraction_bits_first_half[k].append(first_half_bits[index])
                    index += 1

            # 从第二半段恢复符号位、指数位和尾数位
            sign_bits_second_half = second_half_bits[:11]
            exponent_bits_second_half = [[] for _ in range(11)]
            fraction_bits_second_half = [[] for _ in range(11)]

            index = 11
            # 恢复第二半部分的指数位
            for j in range(8):
                for k in range(11):
                    exponent_bits_second_half[k].append(second_half_bits[index])
                    index += 1

            # 恢复第二半部分的尾数位
            for j in range(7):
                for k in range(11):
                    fraction_bits_second_half[k].append(second_half_bits[index])
                    index += 1

            # 组合符号位、指数位和尾数位（第一半）
            for j in range(11):
                recovered_bits = []
                sign = sign_bits_first_half[j]
                exponent = exponent_bits_first_half[j]
                fraction = fraction_bits_first_half[j]
                recovered_bits.append(sign)
                recovered_bits.extend(exponent)
                recovered_bits.extend(fraction)
                recovered_bytes = pack_bits_to_bytes(recovered_bits)
                
                f.write(struct.pack('B', recovered_bytes[1]))
                f.write(struct.pack('B', recovered_bytes[0]))


            # 组合符号位、指数位和尾数位（第二半）
            for j in range(11):
                recovered_bits = []
                sign = sign_bits_second_half[j]
                exponent = exponent_bits_second_half[j]
                fraction = fraction_bits_second_half[j]
                recovered_bits.append(sign)
                recovered_bits.extend(exponent)
                recovered_bits.extend(fraction)
                recovered_bytes = pack_bits_to_bytes(recovered_bits)
                f.write(struct.pack('B', recovered_bytes[1]))
                f.write(struct.pack('B', recovered_bytes[0]))


def handle_exit(executor):
    """Handle graceful shutdown of process pool on exit signals"""
    print("Shutting down executor")
    executor.shutdown(wait=True)
    sys.exit(0)


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
    binary_interleave_dir = root
    binary_compress_dir = os.path.join(root, "compression")
    offcut_dir = os.path.join(root, "offcut")

    read_bfloat16_binary_interleave_bits(binary_interleave_dir+'/model.safetensors.part.121', "/tmp/model.tensor", segment_size=22)

if __name__ == "__main__":
    main()