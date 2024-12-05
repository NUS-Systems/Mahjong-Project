import argparse
import math
import os

def generate_host_tasks(hosts_file, total_tasks, output_dir, output_filename):
    # Read the hosts from the file
    with open(hosts_file, 'r') as f:
        lines = f.readlines()

    # Filter out the host entries that are not comments or empty lines
    hosts = [line.strip() for line in lines if line.strip() and not line.startswith('[') and not line.startswith('#')]

    # Calculate the number of tasks per host
    tasks_per_host = math.ceil(total_tasks / len(hosts))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the host_tasks.yml content
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write("host_specific_tasks:\n")
        for index, host in enumerate(hosts):
            start_task = index * tasks_per_host
            end_task = min(start_task + tasks_per_host, total_tasks)
            f.write(f"  {host}:\n")
            f.write(f"    range: \"{start_task}-{end_task-1}\"\n")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate host_tasks.yml from a hosts file.')
    parser.add_argument('hosts_file', help='The path to the Ansible hosts file.')
    parser.add_argument('total_tasks', type=int, help='The total number of tasks to distribute.')
    parser.add_argument('output_dir', help='The directory where the output file will be saved.')
    parser.add_argument('output_filename', help='The name of the output YAML file.')

    # Parse the arguments
    args = parser.parse_args()

    # Generate the host tasks
    generate_host_tasks(args.hosts_file, args.total_tasks, args.output_dir, args.output_filename)

if __name__ == '__main__':
    main()
