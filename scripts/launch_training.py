import subprocess
import sys
import os
import time
import pandas as pd
import argparse

def create_tmux_window(gpu_id, command):
    # Check if tmux session exists, create one if it doesn't
    result = subprocess.run(['tmux', 'has-session'], capture_output=True)
    if result.returncode != 0:
        print("No tmux session found, creating new session...")
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'training'])
    
    window_name = f"{command.split(' ')[-1].split('/')[-1].split('.')[0]}"
    subprocess.run(['tmux', 'new-window', '-n', window_name])
    full_command = "source ~/.zshrc && conda activate oxe_auge_sim_exp && cd robomimic-mirage/robomimic/scripts"
    full_command += f" && CUDA_VISIBLE_DEVICES={gpu_id} {command}; read -p 'Press Enter to close...'"
    subprocess.run(['tmux', 'send-keys', '-t', window_name, full_command, 'C-m'])
    
    # List all tmux windows

def launch_training(gpu_list, start_index, end_index):
    gpu_idx = 0
    commands = []
    with open('commands/training_commands.txt', 'r') as f:
        commands = f.readlines()

    for index, command in enumerate(commands):
        if index >= start_index and index <= end_index:
            gpu_id = gpu_list[gpu_idx % len(gpu_list)]
            create_tmux_window(gpu_id, command)
        gpu_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch training jobs across multiple GPUs')
    parser.add_argument('--gpus', type=int, nargs='+', required=True, 
                       help='List of GPU IDs to use for training (e.g., --gpus 0 1 2 3)')
    parser.add_argument('--start_index', type=int, required=True, 
                       help='Start index for the commands (0 indexed, inclusive)')
    parser.add_argument('--end_index', type=int, required=True, 
                       help='End index for the commands (0 indexed, inclusive)')
    args = parser.parse_args()
    gpus = args.gpus
    launch_training(gpus, args.start_index, args.end_index)