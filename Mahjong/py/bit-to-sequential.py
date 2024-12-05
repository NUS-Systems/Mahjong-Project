import os
from pathlib import Path
import argparse
import struct
import torch
from safetensors import safe_open

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


def process_folder(input_dir, output_dir, segment_size=4):
    """遍历输入目录下的所有文件并执行转换"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise ValueError("输入路径必须是一个有效的文件夹")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.iterdir():
        if file.is_file():
            output_file = output_path / file.name
            # print(f"处理文件: {file} -> {output_file}")
            read_bfloat16_binary_interleave_bits(file, output_file, segment_size)


def main():
    parser = argparse.ArgumentParser(description="批量处理文件夹中的文件")
    parser.add_argument('input_dir', help='输入文件夹路径')
    parser.add_argument('output_dir', help='输出文件夹路径')
    parser.add_argument('--segment_size', type=int, default=22, help='分段大小（默认: 22）')

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.segment_size)


if __name__ == "__main__":
    main()
