import h5py
import argparse
import numpy as np
import os
import shutil
import json
lengths = {
    'can': 200,
    'square': 200,
    'lift':200,
    'stack':1000,
    'two_piece_assembly':1000
}
robots = ['Jaco', 'Panda',
        'Sawyer',
        'UR5e',
        'Kinova3']
# Define blacklisted trajectories for each robot and task (not actually used)
blacklist_task = { task: {
        'Jaco': [],
        'Panda': [],
        'Sawyer': [],
        'UR5e': [],
        'Kinova3': []
    } for task in ['can', 'lift', 'square', 'stack', 'two_piece_assembly']}


def split_robot_data(input_hdf5_path, output_dir, task):

    os.makedirs(output_dir, exist_ok=True)    
    # Create separate HDF5 files for each robot
    for robot_id in range(len(robots)):
        output_path = os.path.join(output_dir, f"robot{robots[robot_id]}.hdf5")
        

        shutil.copy2(input_hdf5_path, output_path)
        

        with h5py.File(output_path, 'r+') as f:

            with h5py.File(input_hdf5_path, 'r') as f_in:
                if 'env_args' in f_in['data'].attrs:
                    env_args = json.loads(f_in['data'].attrs['env_args'])
                    f['data'].attrs['env_args'] = json.dumps(env_args)
                
                # Copy mask data
                                
                if 'mask' not in f:
                    f.create_group('mask')

                mask_name = 'train'
                keep_keys = [
                    f'demo_{i}' for i in range(lengths[task])
                    if i not in blacklist_task[task][robots[robot_id]]
                ]
                if f.get(f'mask/{mask_name}'):
                    del f[f'mask/{mask_name}']
                f.create_dataset(f'mask/{mask_name}', data=[k.encode("utf-8") for k in keep_keys])
            
            # Get all demo groups
            demo_groups = [k for k in f['data'].keys() if k.startswith('demo_')]
            
            for demo_group in demo_groups:
                # Copy num_samples attribute for each episode
                if 'num_samples' in f[f'data/{demo_group}'].attrs:
                    f[f'data/{demo_group}'].attrs['num_samples'] = f[f'data/{demo_group}'].attrs['num_samples']
                
                obs_group = f[f'data/{demo_group}/obs']
                
                # Get all agentview_image and eef_error datasets
                agentview_images = [k for k in obs_group.keys() if 'agentview_image' in k.lower()]
                
                # Keep only the datasets for this robot
                for dataset_name in agentview_images:
                    if robots[robot_id].lower() not in dataset_name.lower():
                        # Delete datasets that don't belong to this robot
                        del obs_group[dataset_name]
                    else:
                        if dataset_name.startswith('agentview_image_'):
                            new_name = 'agentview_image'
                        else:
                            continue
                            
                        obs_group[new_name] = obs_group[dataset_name]
                        del obs_group[dataset_name]
                
                # Add blacklist attribute based on task and robot
                demo_num = int(demo_group.split('_')[-1])
                is_blacklisted = demo_num in blacklist_task[task][robots[robot_id]]
                f[f'data/{demo_group}'].attrs['blacklist'] = is_blacklisted
                # Store original demo number
                f[f'data/{demo_group}'].attrs['original_demo_num'] = demo_num
        
        print(f"Created split data for robot {robots[robot_id]} at: {output_path}")

def convert_hdf5(input_path, output_path, task):

    with h5py.File(input_path, 'r') as f_in:

        with h5py.File(output_path, 'w') as f_out:

            data_group = f_out.create_group('data')
            
            # Copy env_args attribute from input file
            if 'env_args' in f_in['data'].attrs:
                env_args = json.loads(f_in['data'].attrs['env_args'])
                data_group.attrs['env_args'] = json.dumps(env_args)
            
            # Copy mask data
            if 'mask' in f_in:
                mask_group = f_out.create_group('mask')
                for mask_name in f_in['mask'].keys():
                    mask_data = f_in[f'mask/{mask_name}'][:]
                    mask_group.create_dataset(mask_name, data=mask_data)
            
            # Get all demo groups
            demos = [k for k in f_in['data'].keys() if k.startswith('demo_')]
            
            for demo in demos:
                demo_group = f_in[f'data/{demo}']
                out_demo_group = data_group.create_group(demo)
                
                if 'num_samples' in demo_group.attrs:
                    out_demo_group.attrs['num_samples'] = demo_group.attrs['num_samples']
                
                for dataset in ['actions', 'dones', 'rewards', 'states']:
                    if dataset in demo_group:
                        data = demo_group[dataset][:]
                        out_demo_group.create_dataset(dataset, data=data)
                obs_group = out_demo_group.create_group('obs')
  
                for robot in robots:
                    try:
                        # Copy agentview image
                        if robot == 'Panda':
                            # For Panda, use the base agentview_image
                            agentview_image = demo_group['obs/agentview_image'][:]
                            obs_group.create_dataset('agentview_image_panda', data=agentview_image)
                        else:
                            # For other robots, use the robot-specific image
                            agentview_image = demo_group[f'obs/agentview_image_{robot}'][:]
                            obs_group.create_dataset(f'agentview_image_{robot.lower()}', data=agentview_image)
                    except KeyError:

                        print(f"Warning: agentview_image for {robot} not found in {demo}")
                

                for dataset in ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                    try:
                        if dataset in demo_group['obs']:
                            data = demo_group[f'obs/{dataset}'][:]
                            obs_group.create_dataset(dataset, data=data)
                    except KeyError:
                        print(f"Warning: {dataset} not found in {demo}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str, required=True)
    args = parser.parse_args()

    for task in ['can', 'lift', 'square', 'stack', 'two_piece_assembly']:
        input_path = f'{args.input_data_path}/{task}/image_84.hdf5'
        tmp_path = f'{args.input_data_path}/{task}/tmp.hdf5' 
        output_path = f'xembody_data/{task}/split_data'
        os.makedirs(output_path, exist_ok=True) 
        convert_hdf5(input_path, tmp_path, task)
        split_robot_data(tmp_path, output_path, task) 
        os.remove(tmp_path)
