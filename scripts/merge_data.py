import h5py
import os
import numpy as np
import json

def merge_hdf5_files(input_files, output_file, task):
    if task in ['can', 'square', 'lift']:
        length = 200
    elif task in ['stack', 'two_piece_assembly']:
        length = 1000
    else:
        assert(False)
    # Create output file
    with h5py.File(output_file, 'w') as f_out:
        # Create data group first
        data_group = f_out.create_group('data')
        
        # Copy env_args and other attributes from the first file
        with h5py.File(input_files[0], 'r') as f_first:
            if 'env_args' in f_first['data'].attrs:
                env_args = json.loads(f_first['data'].attrs['env_args'])
                data_group.attrs['env_args'] = json.dumps(env_args)
                
        keep_keys = []

        for file_ind in range(len(input_files)):
            with h5py.File(input_files[file_ind], 'r') as f:                
                mask_name = 'train'
                mask_data = [b.decode('utf-8') for b in list(f[f'mask/{mask_name}'])]
                keep_keys.extend([f"demo_{file_ind * length + int(s.split('_')[1])}" for s in mask_data])
        f_out.create_dataset(f'mask/{mask_name}', data=[k.encode("utf-8") for k in keep_keys])
        print(keep_keys)
        demo_count = 0
        
        for input_file in input_files:
            print(f"Processing {input_file}...")
            with h5py.File(input_file, 'r') as f_in:
                # Get all demo groups
                demo_groups = [k for k in f_in['data'].keys() if k.startswith('demo_')]
                demo_groups.sort(key=lambda x: int(x.split('_')[1]))
                
                for demo in demo_groups:
                    # Create new demo name with sequential number
                    new_demo = f"demo_{demo_count}"
                    demo_count += 1
                    
                    # Create corresponding group in output
                    out_demo_group = f_out.create_group(f'data/{new_demo}')
                    
                    # Copy num_samples attribute
                    if 'num_samples' in f_in[f'data/{demo}'].attrs:
                        out_demo_group.attrs['num_samples'] = f_in[f'data/{demo}'].attrs['num_samples']
                    
                    # Copy all datasets from the demo group
                    for dataset_name, dataset in f_in[f'data/{demo}'].items():
                        
                        if isinstance(dataset, h5py.Dataset):
                            # Copy dataset
                            f_out.create_dataset(f'data/{new_demo}/{dataset_name}', 
                                               data=dataset[:],
                                               dtype=dataset.dtype)
                        elif isinstance(dataset, h5py.Group):
                            # Create subgroup and copy its datasets
                            subgroup = f_out.create_group(f'data/{new_demo}/{dataset_name}')
                            for subdataset_name, subdataset in dataset.items():
                                if 'eef_error' in subdataset_name: continue
                                subgroup.create_dataset(subdataset_name,
                                                      data=subdataset[:],
                                                      dtype=subdataset.dtype)
                    
                    # Copy all other attributes
                    for attr_name, attr_value in f_in[f'data/{demo}'].attrs.items():
                        if attr_name != 'num_samples':  # We already copied this above
                            f_out[f'data/{new_demo}'].attrs[attr_name] = attr_value
                    
                    # Add source file and original demo name as attributes
                    f_out[f'data/{new_demo}'].attrs['source_file'] = os.path.basename(input_file)
                    f_out[f'data/{new_demo}'].attrs['original_demo'] = demo
        print(f"\nMerged {demo_count} demos into {output_file}")

if __name__ == "__main__":
    # List of input files
    for task in ['can', 'lift', 'square', 'stack', 'two_piece_assembly']:
        robot_to_path = {
            'Jaco': f"xembody_data/{task}/split_data/robotJaco.hdf5",
            'Panda': f"xembody_data/{task}/split_data/robotPanda.hdf5",
            'Sawyer': f"xembody_data/{task}/split_data/robotSawyer.hdf5",
            'UR5e': f"xembody_data/{task}/split_data/robotUR5e.hdf5",
            'Kinova3': f"xembody_data/{task}/split_data/robotKinova3.hdf5"
        }
        all = ['Jaco', 'Panda', 'Sawyer', 'UR5e', 'Kinova3']
        dct = {'all_minus_jaco': [i for i in all if i!='Jaco'],
         'all_minus_sawyer': [i for i in all if i!='Sawyer'],
         'all_minus_ur5e': [i for i in all if i!='UR5e'],
         'all_minus_kinova3': [i for i in all if i!='Kinova3']
        }
        #all minus data
        for name in dct.keys():
            input_files = [robot_to_path[robot] for robot in dct[name]]
            assert(len(input_files) == 4)
            os.makedirs(f'xembody_data/{task}/merged_data', exist_ok=True)
            output_file = f"xembody_data/{task}/merged_data/{name}_merged.hdf5".lower()
            merge_hdf5_files(input_files, output_file, task) 
        

        #1+1 and all data
        for robots_to_merge in [['Panda', 'Jaco'], 
                                ['Panda', 'Sawyer'], 
                                ['Panda', 'UR5e'],
                                ['Panda', 'Kinova3'],
                                list(robot_to_path.keys())]:
            input_files = [robot_to_path[robot] for robot in robots_to_merge]
            
            # Output file
            output_file = f"xembody_data/{task}/merged_data/{'_'.join(robots_to_merge)}_merged.hdf5".lower()
            
            merge_hdf5_files(input_files, output_file, task) 
