#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path



def read_commands(mode):
    """Read commands from the appropriate eval commands file based on mode"""
    commands_file = Path(__file__).parent.parent / "commands" / f"{mode}_eval_commands.txt"
    
    if not commands_file.exists():
        raise FileNotFoundError(f"Commands file not found: {commands_file}")
    
    commands = []
    with open(commands_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                commands.append(line)
    
    return commands

def check_tmux_session(session_name):
    """Check if tmux session exists"""
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', session_name], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def create_tmux_session(session_name):
    """Create a new tmux session"""
    if check_tmux_session(session_name):
        print(f"Tmux session '{session_name}' already exists. Killing it...")
        subprocess.run(['tmux', 'kill-session', '-t', session_name])
        time.sleep(1)
    
    print(f"Creating tmux session: {session_name}")
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])

def run_command_in_tmux_window(session_name, window_name, command, gpu_id):
    """Run a command in a new tmux window with proper environment setup"""
    
    # Create new window
    subprocess.run(['tmux', 'new-window', '-t', session_name, '-n', window_name])
    
    # Setup environment: source zshrc, activate conda env, cd to correct directory, set GPU
    setup_commands = [
        'source ~/.zshrc',
        'conda activate oxe_auge_sim_exp',
        'cd mirage/mirage/benchmark/robosuite',
        f'export CUDA_VISIBLE_DEVICES={gpu_id}',
        command
    ]
    
    # Send all commands to the window
    for cmd in setup_commands:
        subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', cmd, 'Enter'])
        time.sleep(0.5)  # Small delay between commands

def main():
    """Main function to orchestrate the tmux evaluation runs"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run evaluation commands in tmux with GPU management')
    parser.add_argument('--mode', choices=['patch', 'lighting', 'standard'], 
                       help='Evaluation mode: patch, lighting, or standard')
    parser.add_argument('--start_index', type=int, required=True, 
                       help='Start index for the commands (0 indexed, inclusive)')
    parser.add_argument('--end_index', type=int, required=True, 
                       help='End index for the commands (0 indexed, inclusive)')
    parser.add_argument('--gpus', type=int, nargs='+', required=True, 
                       help='List of GPU IDs to use for evaluation (e.g., --gpus 0 1 2 3)')
    
    args = parser.parse_args()
    
    try:

        # Read commands from file
        commands = read_commands(args.mode)
        # Create tmux session
        session_name = f"{args.mode}_evals"
        create_tmux_session(session_name)
        
        # Run each command in its own tmux window
        for i, command in enumerate(commands, 1):
            if i >= args.start_index and i <= args.end_index:
                gpu_id = args.gpus[i % len(args.gpus)]
                window_name = f"eval_{i}"
                run_command_in_tmux_window(session_name, window_name, command, gpu_id)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()