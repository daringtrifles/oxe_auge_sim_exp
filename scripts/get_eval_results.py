#!/usr/bin/env python3

import numpy as np
import os
import sys
import csv
import json
import argparse
from pathlib import Path

def find_results(path):
    """Returns num_rollouts, num_success"""
    try:
        with open(path, 'r') as f:
            content = f.read().strip()
            # Split content into individual JSON objects
            json_objects = content.split('}{')
            if len(json_objects) > 1:
                # Add back the missing braces
                json_objects = [json_objects[0] + '}'] + ['{' + obj for obj in json_objects[1:]]
            
            # Parse the first JSON object which contains the main results
            data = json.loads(json_objects[0])
            num_rollouts = data.get('Num Rollouts', 0)
            num_success = data.get('Num_Success', [0])[0]
            return num_rollouts, num_success
    except FileNotFoundError:
        return 0, 0
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {path}. This may be because the first rollout is still running.")
        return 0, 0
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
        return 0, 0

# Robot name mapping
ROBOT_MAP = {'panda': 'Panda', 'sawyer': 'Sawyer', 'ur5e': 'UR5e', 'kinova3': 'Kinova3', 'jaco': 'Jaco', 'iiwa': 'IIWA'}

def test_results(exp_name, mode):
    """Get results for a specific experiment and mode"""
    results = []
    with open('experiments_to_run.csv', 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row['exp_name'] == exp_name:
                # Get all non-None robots for this experiment
                robots = [robot for robot in [row['eval_robot_1'], row['eval_robot_2'], row['eval_robot_3'], 
                                            row['eval_robot_4'], row['eval_robot_5']] 
                         if robot and robot.lower() != 'nan']
                
                for robot in robots:
                    robot_name = ROBOT_MAP[robot.lower()]
                    if mode in ['patch', 'lighting'] and robot_name != 'Panda':
                        continue
                    results_path = get_results_path(exp_name, robot_name, mode)
                    num_rollouts, num_success = find_results(results_path)
                    
                    results.append({
                        'Exp Name': exp_name,
                        'Robot': robot_name,
                        'Mode': mode,
                        'Num Rollouts': num_rollouts,
                        'Num Success': num_success,
                        'Success Rate': num_success/num_rollouts if num_rollouts > 0 else 0,
                    })
    return results

def get_results_path(exp_name, robot, mode):
    """Get the path to results file for given experiment, robot, and mode"""
    return f'mirage/mirage/benchmark/robosuite/results/{exp_name}/{robot}_{mode}/target.txt'

def get_experiment_list():
    """Get the list of experiments to run (matching create_mirage_configs.py)"""
    return [
        # Can experiments
        'panda_can', 'sawyer_can', 'jaco_can', 'ur5e_can', 'kinova3_can',
        'panda_sawyer_can', 'panda_jaco_can', 'panda_ur5e_can', 'panda_kinova3_can',
        'all_minus_jaco_merged_can', 'all_minus_kinova3_merged_can', 
        'all_minus_sawyer_merged_can', 'all_minus_ur5e_merged_can', 'all_can',
        
        # Square experiments
        'panda_square', 'sawyer_square', 'jaco_square', 'ur5e_square', 'kinova3_square',
        'panda_sawyer_square', 'panda_jaco_square', 'panda_ur5e_square', 'panda_kinova3_square',
        'all_minus_jaco_merged_square', 'all_minus_kinova3_merged_square',
        'all_minus_sawyer_merged_square', 'all_minus_ur5e_merged_square', 'all_square',
        
        # Stack experiments
        'panda_stack', 'sawyer_stack', 'jaco_stack', 'ur5e_stack', 'kinova3_stack',
        'panda_sawyer_stack', 'panda_jaco_stack', 'panda_ur5e_stack', 'panda_kinova3_stack',
        'all_minus_jaco_merged_stack', 'all_minus_kinova3_merged_stack',
        'all_minus_sawyer_merged_stack', 'all_minus_ur5e_merged_stack', 'all_stack',
        
        # Lift experiments
        'panda_lift', 'sawyer_lift', 'jaco_lift', 'ur5e_lift', 'kinova3_lift',
        'panda_sawyer_lift', 'panda_jaco_lift', 'panda_ur5e_lift', 'panda_kinova3_lift',
        'all_minus_jaco_merged_lift', 'all_minus_kinova3_merged_lift',
        'all_minus_sawyer_merged_lift', 'all_minus_ur5e_merged_lift', 'all_lift',
        
        # Two piece assembly experiments
        'panda_two_piece_assembly', 'sawyer_two_piece_assembly', 'jaco_two_piece_assembly',
        'ur5e_two_piece_assembly', 'kinova3_two_piece_assembly', 'panda_sawyer_two_piece_assembly',
        'panda_jaco_two_piece_assembly', 'panda_ur5e_two_piece_assembly', 'panda_kinova3_two_piece_assembly',
        'all_minus_jaco_merged_two_piece_assembly', 'all_minus_kinova3_merged_two_piece_assembly',
        'all_minus_sawyer_merged_two_piece_assembly', 'all_minus_ur5e_merged_two_piece_assembly',
        'all_two_piece_assembly'
    ]

def main():
    """Main function to collect and save evaluation results"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect evaluation results by mode')
    parser.add_argument('--mode', nargs='?', choices=['patch', 'lighting', 'standard', 'all'], 
                       default='all', help='Evaluation mode: patch, lighting, standard, or all (default: all)')    
    args = parser.parse_args()
    
    # Determine which modes to process
    if args.mode == 'all':
        modes_to_process = ['standard', 'patch', 'lighting']
    else:
        modes_to_process = [args.mode]
    
    experiments = get_experiment_list()
    
    
    
    # Process each mode and experiment combination
    for mode in modes_to_process:
        all_results = []
        for exp_name in experiments:
            # Skip certain combinations based on create_mirage_configs.py logic
            if mode in ['patch', 'lighting'] and 'all_minus' in exp_name:
                continue
                
            results = test_results(exp_name, mode)
            all_results.extend(results)
            
        output_filename = f'evaluation_results_{mode}_mode.csv'
    
        # Write results to CSV
        fieldnames = ['Exp Name', 'Robot', 'Mode', 'Num Rollouts', 'Num Success', 'Success Rate']
        os.makedirs('results', exist_ok=True)
        with open(f'results/{output_filename}', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

if __name__ == "__main__":
    main()