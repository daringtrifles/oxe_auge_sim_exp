
import os
import sys
import csv
import yaml
from pathlib import Path

# Robot name mapping
ROBOT_MAP = {
    'panda': 'Panda', 'sawyer': 'Sawyer', 'ur5e': 'UR5e', 
    'kinova3': 'Kinova3', 'jaco': 'Jaco', 'iiwa': 'IIWA'
}
standard_training_commands = []
patch_training_commands = []
lighting_training_commands = []

def process_experiment(exp_name, mode):
    """Process a single experiment by finding robots and creating configs."""

    with open('experiments_to_run.csv', encoding='utf-8', mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['exp_name'] == exp_name:
                # Get all valid robots for this experiment
                robots = [robot for robot in [
                    row['eval_robot_1'], row['eval_robot_2'], row['eval_robot_3'], 
                    row['eval_robot_4'], row['eval_robot_5']
                ] if robot and robot.lower() != 'nan']
                
                # Process each robot (currently filtered to Panda only)
                for robot in robots:
                    robot_name = ROBOT_MAP[robot.lower()]
                    if mode in ['patch', 'lighting']:
                        if robot_name != 'Panda': 
                            continue
                    create_config(exp_name, robot_name, mode)
                return
        


def get_absolute_path(relative_path):
    """Convert relative path to absolute path."""
    return str(os.path.abspath(relative_path))

def get_model_path(experiment_name):
    """Find the epoch_500 model file for the given experiment."""
    folder_path = f'robomimic-mirage/trained_diffusion_policies/{experiment_name}'
    try:
        files = list_all_files(folder_path)
        for file_path in files:
            if 'epoch_500' in file_path:
                return file_path
        print(f"Warning: No epoch_500 model found for {experiment_name}")
        return None
    except Exception as e:
        print(f"Error finding model for {experiment_name}: {e}")
        return None

def create_config(exp_name, robot, mode):
    """Create configuration file for the experiment."""
    model_path = get_absolute_path(get_model_path(exp_name))
    if not model_path:
        print(f"Error: No model found for experiment '{exp_name}'")
        return
    
    results_folder = get_absolute_path(f'mirage/mirage/benchmark/robosuite/results/{exp_name}/{robot}_{mode}/')
    output_path = get_absolute_path(f'mirage/mirage/benchmark/robosuite/config/{exp_name}/{robot}_{mode}/{robot}.yaml')
    
    config = {
        'source_agent_path': model_path,
        'target_agent_path': model_path,
        'naive': True,
        'n_rollouts': 100,
        'horizon': 350,
        'seed': 0,
        'passive': True,
        'connection': True,
        'source_robot_name': "Panda",
        'target_robot_name': robot,
        'source_tracking_error_threshold': 0.015,
        'source_num_iter_max': 300,
        'target_tracking_error_threshold': 0.015,
        'target_num_iter_max': 300,
        'delta_action': False,
        'enable_inpainting': False,
        'use_ros': False,
        'offline_eval': False,
        'use_diffusion': False,
        'diffusion_input_type': "",
        'results_folder': results_folder,
        'target_video_path': None,
        'source_video_path': None,
        'add_patches': mode == 'patch'
    }
    

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, encoding='utf-8', mode='w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    if mode == 'standard':
        standard_training_commands.append(f"python run_robosuite_benchmark.py --config {output_path}")
    elif mode == 'patch':
        patch_training_commands.append(f"python run_robosuite_benchmark.py --config {output_path}")
    elif mode == 'lighting':
        lighting_training_commands.append(f"python run_robosuite_benchmark.py --config {output_path}")
    
    Path(results_folder + '/source.txt').touch()
    Path(results_folder + '/target.txt').touch()
        

def list_all_files(directory):
    """Recursively list all files in a directory."""
    file_list = []
    try:
        entries = sorted(os.listdir(directory))
        for entry in entries:
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                file_list.extend(list_all_files(full_path))
            else:
                file_list.append(full_path)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing '{directory}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return file_list

def get_experiment_list():
    """Get the list of experiments to run."""
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

    experiments = get_experiment_list()
    
    for exp_name in experiments:
        for mode in ['lighting', 'patch', 'standard']:
            if mode in ['patch', 'lighting']:
                if 'all_minus' in exp_name:
                    continue
            process_experiment(exp_name, mode)
    os.makedirs('commands', exist_ok=True)
    with open('commands/standard_eval_commands.txt', encoding='utf-8', mode='w') as f:
        for command in standard_training_commands:
            f.write(command + '\n')
    with open('commands/patch_eval_commands.txt', encoding='utf-8', mode='w') as f:
        for command in patch_training_commands:
            f.write(command + '\n')
    with open('commands/lighting_eval_commands.txt', encoding='utf-8', mode='w') as f:
        for command in lighting_training_commands:
            f.write(command + '\n')

if __name__ == "__main__":
    main()