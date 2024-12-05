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

def extract_payload(file_path):
    """提取除去metadata的payload部分"""
    with open(file_path, 'rb') as f:
        # 读取metadata长度（假设前8字节表示metadata的长度）
        metadata_length_bytes = f.read(8)
        metadata_length = int.from_bytes(metadata_length_bytes, 'little')
        
        # 跳过metadata部分
        f.seek(8 + metadata_length)
        
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
    # 将 tensor 转换为 bfloat16 类型
    tensor_bf16 = tensor.to(torch.bfloat16)
    with open(file_path, 'wb') as f:
        # 直接将 bfloat16 作为 16 位无符号整数写入文件
        for value in tensor_bf16:
            binary_value = struct.pack('>H', value.view(torch.uint16).item())  # 大端序
            f.write(binary_value)

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

def write_binary_interleave_bits(input_sequential_file, file_path, segment_size=44):
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

    payload = b''
    with open(input_sequential_file, 'rb') as f:
        payload = f.read()
    
    segment_size = segment_size*2
    with open(file_path, 'wb') as f:
        for i in range(0, len(payload), segment_size):
            segment = payload[i:i + segment_size]
            half_size = len(segment) // 2

            # 第一半部分的符号位、指数位和尾数位
            sign_bits_first_half = []
            exponent_bits_first_half = [[] for _ in range(8)]  # 8 位指数
            fraction_bits_first_half = [[] for _ in range(7)]  # 7 位尾数

            # 第二半部分的符号位、指数位和尾数位
            sign_bits_second_half = []
            exponent_bits_second_half = [[] for _ in range(8)]  # 8 位指数
            fraction_bits_second_half = [[] for _ in range(7)]  # 7 位尾数

            # 分割 segment 为两部分
            first_half = segment[:half_size]
            second_half = segment[half_size:]

            # 处理第一半部分
            for v in range(0, len(first_half), 2):
                
                binary_value = int.from_bytes(first_half[v:v+2], byteorder='little')
                sign = (binary_value >> 15) & 0x1
                exponent = (binary_value >> 7) & 0xFF  # 8 位指数
                fraction = binary_value & 0x7F  # 7 位尾数

                sign_bits_first_half.append(sign)

                # 分解指数位（8 位）
                exponent_bits_per_value = [int(b) for b in format(exponent, '08b')]
                for j in range(8):
                    exponent_bits_first_half[j].append(exponent_bits_per_value[j])

                # 分解尾数位（7 位）
                fraction_bits_per_value = [int(b) for b in format(fraction, '07b')]
                for j in range(7):
                    fraction_bits_first_half[j].append(fraction_bits_per_value[j])

            # 处理第二半部分（顺序提取，与第一半部分相同的方式）
            for v in range(0, len(second_half), 2):
                binary_value = int.from_bytes(second_half[v:v+2], byteorder='little')
                sign = (binary_value >> 15) & 0x1
                exponent = (binary_value >> 7) & 0xFF  # 8 位指数
                fraction = binary_value & 0x7F  # 7 位尾数

                sign_bits_second_half.append(sign)

                # 分解指数位（8 位）
                exponent_bits_per_value = [int(b) for b in format(exponent, '08b')]
                for j in range(8):
                    exponent_bits_second_half[j].append(exponent_bits_per_value[j])

                # 分解尾数位（7 位）
                fraction_bits_per_value = [int(b) for b in format(fraction, '07b')]
                for j in range(7):
                    fraction_bits_second_half[j].append(fraction_bits_per_value[j])

            # 汇总交错后的位集合
            interleaved_bits = []

            # 第一部分：符号位 -> 指数位 -> 尾数位
            interleaved_bits.extend(sign_bits_first_half)
            for j in range(8):
                interleaved_bits.extend(exponent_bits_first_half[j])
            for j in range(7):
                interleaved_bits.extend(fraction_bits_first_half[j])

            # 第二部分：符号位 -> 指数位 -> 尾数位
            # 先提取所有位，然后对整个数组进行反转
            second_half_bits = []

            # 收集第二部分的位
            second_half_bits.extend(sign_bits_second_half)
            for j in range(8):
                second_half_bits.extend(exponent_bits_second_half[j])
            for j in range(7):
                second_half_bits.extend(fraction_bits_second_half[j])

            # 对第二部分的所有位进行反转
            second_half_bits.reverse()

            # 将反转后的第二部分位添加到 interleaved_bits 中
            interleaved_bits.extend(second_half_bits)

            # 打包为字节并写入文件
            interleaved_bytes = pack_bits_to_bytes(interleaved_bits)
            for byte in interleaved_bytes:
                f.write(struct.pack('B', byte))

# 恢复函数：从文件反向读取交错存储的符号位、指数位和尾数位 (bfloat16)
def read_bfloat16_binary_interleave_bits(file_path, tensor_size, segment_size=4):
    recovered_tensor = []  # 存储恢复的bfloat16值

    with open(file_path, 'rb') as f:
        for i in range(0, tensor_size, segment_size):
            current_segment_size = min(segment_size, tensor_size - i)
            half_size = current_segment_size // 2

            # 计算需要读取的字节数
            bit_count = current_segment_size * (1 + 8 + 7)  # 每个数 1 位符号，8 位指数，7 位尾数
            byte_count = (bit_count + 7) // 8  # 除以 8 并向上取整
            interleaved_bytes = f.read(byte_count)
            interleaved_bits = unpack_bytes_to_bits(interleaved_bytes)

            # 分割位数组为前半段和后半段
            first_half_bits = interleaved_bits[:half_size * (1 + 8 + 7)]
            second_half_bits = interleaved_bits[half_size * (1 + 8 + 7):]

            # 对第二半段的位进行反转（恢复原始顺序）
            second_half_bits.reverse()

            # 从前半段恢复符号位、指数位和尾数位
            sign_bits_first_half = first_half_bits[:half_size]
            exponent_bits_first_half = [[] for _ in range(half_size)]
            fraction_bits_first_half = [[] for _ in range(half_size)]

            index = half_size

            # 恢复第一半部分的指数位
            for j in range(8):
                for k in range(half_size):
                    exponent_bits_first_half[k].append(first_half_bits[index])
                    index += 1

            # 恢复第一半部分的尾数位
            for j in range(7):
                for k in range(half_size):
                    fraction_bits_first_half[k].append(first_half_bits[index])
                    index += 1

            # 从第二半段恢复符号位、指数位和尾数位
            sign_bits_second_half = second_half_bits[:half_size]
            exponent_bits_second_half = [[] for _ in range(half_size)]
            fraction_bits_second_half = [[] for _ in range(half_size)]

            index = half_size

            # 恢复第二半部分的指数位
            for j in range(8):
                for k in range(half_size):
                    exponent_bits_second_half[k].append(second_half_bits[index])
                    index += 1

            # 恢复第二半部分的尾数位
            for j in range(7):
                for k in range(half_size):
                    fraction_bits_second_half[k].append(second_half_bits[index])
                    index += 1

            # 组合符号位、指数位和尾数位（第一半）
            for j in range(half_size):
                sign = sign_bits_first_half[j]
                exponent = int(''.join(map(str, exponent_bits_first_half[j])), 2)
                fraction = int(''.join(map(str, fraction_bits_first_half[j])), 2)

                # 组合为 BFP16 的 16 位整数
                binary_bf16 = (sign << 15) | (exponent << 7) | fraction
                value = torch.tensor([binary_bf16], dtype=torch.uint16).view(torch.bfloat16).item()
                recovered_tensor.append(value)

            # 组合符号位、指数位和尾数位（第二半）
            for j in range(half_size):
                sign = sign_bits_second_half[j]
                exponent = int(''.join(map(str, exponent_bits_second_half[j])), 2)
                fraction = int(''.join(map(str, fraction_bits_second_half[j])), 2)

                # 组合为 BFP16 的 16 位整数
                binary_bf16 = (sign << 15) | (exponent << 7) | fraction
                value = torch.tensor([binary_bf16], dtype=torch.uint16).view(torch.bfloat16).item()
                recovered_tensor.append(value)

    return torch.tensor(recovered_tensor, dtype=torch.bfloat16)

# 验证恢复是否正确
def verify_recovery(original_tensor, recovered_tensor):
    return torch.allclose(original_tensor, recovered_tensor, atol=1e-3)  # 允许一定的浮点误差

def compress_bit_interleave(input_file, output_file):
    segment_size = 44  # 每个segment的大小

    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        while True:
            # 读取一个segment
            segment = f_in.read(segment_size)
            if len(segment) < segment_size:
                break  # 如果segment大小不足，表示文件末尾，退出循环

            # 拿出需要挪动的部分
            front = segment[:9]  # 第3到13字节
            end = segment[-9:]  # 第74到75字节
            middle = segment[9:35]

            middle_left = middle[:13]
            middle_right = middle[13:]

            new_segment = middle_left + front + end + middle_right

            # 最终segment的长度应该是88字节
            assert len(new_segment) == segment_size, f"Segment size mismatch: expected {segment_size}, got {len(new_segment)}"

            # 将新的segment写入输出文件
            f_out.write(new_segment)

def process_write_binary_interleave(file_path, file_name, binary_interleave_dir, segment_size):
    if not os.path.exists(binary_interleave_dir):
        os.makedirs(binary_interleave_dir, exist_ok=True)
    binary_interleave_file = os.path.join(binary_interleave_dir, file_name)
    write_binary_interleave_bits(file_path, binary_interleave_file, segment_size=segment_size)

def process_compress_binary(binary_interleave_file, binary_compress_file):
    if not os.path.exists(os.path.dirname(binary_compress_file)):
        os.makedirs(os.path.dirname(binary_compress_file), exist_ok=True)
    compress_bit_interleave(binary_interleave_file, binary_compress_file)

def process_file(file_path, binary_interleave_dir, binary_compress_dir, number_on_strand):
    try:
    # 加载张量
        file_name = os.path.basename(file_path)

    # # 顺序执行每个任务
    # # 1. 处理 write_component_interleave
    #     process_write_component(tensor, file_name, component_interleave_dir, number_on_strand)

    # 2. 处理 write_binary_interleave_bits
        process_write_binary_interleave(file_path, file_name, binary_interleave_dir, number_on_strand)

    # 3. 在 write_binary_interleave_bits 完成后，进行压缩
        binary_interleave_file = os.path.join(binary_interleave_dir, file_name)
        binary_compress_file = os.path.join(binary_compress_dir, file_name)
        process_compress_binary(binary_interleave_file, binary_compress_file)
    except Exception as e:
        print(f"处理任务的error {e}")


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
    binary_interleave_dir = os.path.join(root, "bit")
    binary_compress_dir = os.path.join(root, "compression")
    offcut_dir = os.path.join(root, "offcut")

    # 运行文件处理和分割
    # process_and_split_file(args.input_file, sequential_dir, offcut_dir, args.chunk_size)

    # 创建进程池
    with ProcessPoolExecutor(max_workers=64) as executor:
        # 注册信号处理函数，以确保可以优雅地关闭进程池
        signal.signal(signal.SIGINT, lambda s, f: handle_exit(executor))
        signal.signal(signal.SIGTERM, lambda s, f: handle_exit(executor))

        # 收集所有要处理的文件
        files_to_process = []
        for root, dirs, files in os.walk(sequential_dir):
            for file_name in files:
                file_path = os.path.join(sequential_dir, file_name)
                files_to_process.append(file_path)

        # 分批次处理任务
        batch_size = args.batch_size  # 每次提交的任务数量
        futures = []

        # 使用迭代器分批次提交任务
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]

            print(f"正在提交新的一批任务，共 {len(batch)} 个任务。")
            
            for file_path in batch:
                futures.append(executor.submit(process_file, file_path,  binary_interleave_dir, binary_compress_dir, number_on_strand))

            # 等待当前一批任务完成
            try:
                for future in as_completed(futures):
                    try:
                        result = future.result()  # 获取任务结果
                    except Exception as e:
                        print(f"处理文件时出现错误: {e}")
            except (KeyboardInterrupt, SystemExit):
                print("接收到中断信号，正在关闭...")
                handle_exit(executor)
                break  # 退出批次提交循环
            finally:
                futures.clear()  # 清空已完成的任务

        print("所有任务完成，关闭进程池。")

if __name__ == "__main__":
    main()