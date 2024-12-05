import os
import re
import argparse

def merge_files(input_dir, output_file, base_file_name):
    # Get all chunk files with the specified base name and sort them
    chunk_files = sorted(f for f in os.listdir(input_dir) if f.startswith(base_file_name))
    
    # Custom sort function to extract the number from the file name
    def extract_chunk_number(file_name):
        match = re.search(r'\.part\.(\d+)$', file_name)
        return int(match.group(1)) if match else -1

    # Sort files based on the extracted number
    chunk_files.sort(key=extract_chunk_number)
    print(f"Merging files: {chunk_files}")

    # Open the output file and write the content of each chunk into it
    with open(output_file, 'wb') as output_f:
        for chunk_file in chunk_files:
            chunk_file_path = os.path.join(input_dir, chunk_file)
            with open(chunk_file_path, 'rb') as f:
                output_f.write(f.read())

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge file chunks into a single file.')
    parser.add_argument('input_dir', type=str, help='Directory containing the chunk files')
    parser.add_argument('output_file', type=str, help='Path for the output merged file')
    parser.add_argument('base_file_name', type=str, help='Base name of the chunk files')

    # Parse arguments
    args = parser.parse_args()

    # Call merge files with parsed arguments
    merge_files(args.input_dir, args.output_file, args.base_file_name)

if __name__ == '__main__':
    main()