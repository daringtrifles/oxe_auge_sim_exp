import json
import pandas as pd
import os
def create_config(name, dataset):
    # Read the template file
    with open('robomimic-mirage/configs/template.json', 'r') as f:
        template = json.load(f)
    
    # Update the algo_name
    template['experiment']['name'] = name
    dataset = f'../../../{dataset}'
    # Update the train/data/path in the template
    # train/data is a list containing dictionaries with path
    if 'train' in template and 'data' in template['train'] and isinstance(template['train']['data'], list):
        if len(template['train']['data']) > 0:
            template['train']['data'][0]['path'] = dataset
        else:
            # If the list is empty, add a new dictionary with the path
            template['train']['data'].append({'path': dataset})
    
    # Create the output filename
    output_file = f"robomimic-mirage/configs/{name}.json"
    
    # Write the new config file
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=4)
    
    print(f"Created new config file: {output_file}")
    return output_file

def main():
    experiments_to_run = pd.read_csv('experiments_to_run.csv')
    commands = []
    for _, row in experiments_to_run.iterrows():
        create_config(row['exp_name'], row['train_data_filepath'])
        command = f"python train.py --config ../../../robomimic-mirage/configs/{row['exp_name']}.json"
        commands.append(command)
    os.makedirs('commands', exist_ok=True)
    with open('commands/training_commands.txt', 'w') as f:
        for command in commands:
            f.write(command + '\n')

if __name__ == "__main__":
    main() 