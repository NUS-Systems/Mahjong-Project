import os
import argparse

def decompress_bit_interleave(input_file, output_file, bit_file):
    segment_size = 44  # 压缩后的每个segment大小

    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out, open(bit_file, 'rb') as f_bit:
        while True:
            # 读取压缩后的segment
            compressed_segment = f_in.read(segment_size)
            interleave_segment = f_bit.read(segment_size)
            if len(compressed_segment) < segment_size:
                break  # 文件末尾

            # 转换为二进制表示
            bit_segment = ''.join(f"{byte:08b}" for byte in compressed_segment)
            bit_interleave_segment = ''.join(f"{byte:08b}" for byte in interleave_segment)
            front_bits = bit_interleave_segment[12:52]  # Next 5 bytes (40 bits)
            end_bits = bit_interleave_segment[-52:-12]  # Last 6.5 bytes except 1.5 bytes

            # 按压缩后的顺序拆分段
            head_bits = bit_segment[:12]  # First 1.5 bytes (12 bits)
            middle_front_bits = bit_segment[12:136]  # Next 15.5 bytes (124 bits)
            # front_bits = bit_segment[136:176]  # Next 5 bytes (40 bits)
            # end_bits = bit_segment[176:-136]  # Next 6.5 bytes (52 bits)
            middle_end_bits = bit_segment[-136:-12]  # Next 15.5 bytes (124 bits)
            tail_bits = bit_segment[-12:]  # Last 1.5 bytes (12 bits)

            # 恢复原始顺序
            original_segment_bits = (
                head_bits + front_bits + middle_front_bits +
                middle_end_bits + end_bits + tail_bits
            )

            # 转换为字节
            original_segment = int(original_segment_bits, 2).to_bytes((len(original_segment_bits) + 7) // 8, byteorder='big')

            # 写入恢复的文件
            f_out.write(original_segment)

# 示例使用
def process_folders(folder1, folder2, output_folder):
    """
    以 folder2 的文件为基准，匹配 folder1 中的文件，处理后输出到 output_folder。
    :param folder1: 第一个文件夹路径 (存储 bit_file)
    :param folder2: 第二个文件夹路径 (存储 input_compressed_file)
    :param output_folder: 输出的文件存储路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历第二个文件夹中的文件
    for root, _, files in os.walk(folder2):
        for file in files:
            input_file = os.path.join(root, file)
            bit_file = os.path.join(folder1, file)
            output_file = os.path.join(output_folder, file)

            # 检查文件是否在 folder1 中存在
            if os.path.exists(bit_file):
                print(f"Processing: {file}")
                try:
                    decompress_bit_interleave(input_file, output_file, bit_file)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
            else:
                print(f"Skipping {file}, matching bit file not found in folder1.")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process files across three folders.")
    parser.add_argument("folder1", help="Path to the first folder (bit files)")
    parser.add_argument("folder2", help="Path to the second folder (compressed files)")
    parser.add_argument("output_folder", help="Path to the output folder")

    args = parser.parse_args()

    # 调用主处理函数
    process_folders(args.folder1, args.folder2, args.output_folder)
