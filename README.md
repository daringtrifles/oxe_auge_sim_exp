# OXE-AugE Simulation

This repository evaluates code using [Mirage](https://github.com/BerkeleyAutomation/mirage), and provides scripts for running the simulation experiments in the OXE-AugE paper.

Website: https://oxe-auge.github.io/

---

## Installation

### 1. Git clone and create the Conda environment
```bash
git clone --recurse git@github.com:daringtrifles/oxe_auge_sim_exp.git
cd oxe_auge_sim_exp
conda env create -f oxe_auge_sim_exp.yaml
conda activate oxe_auge_sim_exp
```

### 2. Install editable packages
```bash
pip install -e mirage/
pip install -e mimicgen_environments/
pip install -e robomimic-mirage/
pip install -e robosuite/
```

---

> If you want to follow along with training, you can run the data preparation and training steps below. Otherwise, you can download checkpoints from [here](https://drive.google.com/drive/folders/1UvvcliDObOwiM-OuNaSSUlmKURrCIMTZ?usp=sharing) and place the `trained_diffusion_policies` folder inside `robomimic-mirage`

## Data Preparation

### To obtain the augmented simulation data, either [follow these steps](https://github.com/GuanhuaJi/rovi-aug-extension-robosuite) to generate the data, or [download the data](https://drive.google.com/drive/folders/1rMnAwPSM_Q3gBWDjwXHsdI2VXmZmx7A9)

After procuring the data, run the following

```bash
bash bash_scripts/prepare_data.sh <PATH_TO_SIMULATION_DATA>
```
>⚠️ Data preparation may take **~1 hour**. It is recommended to run this step inside a `tmux` session.
---

## Training
### 1. Generate training configs
```bash
python scripts/create_training_configs.py
```

### 2. Switch file utilities for training
```bash
python scripts/switch_file_utils.py train
```

### 3. Launch training
#### Option 1: Launch via script
```bash
python scripts/launch_training.py --gpus <GPU_IDs> --start_index <START_INDEX> --end_index <END_INDEX>
```

#### Option 2: Launch commands manually

You can find the commands in `commands/training_commands.txt`.  Make sure to run them inside the `robomimic-mirage/robomimic/scripts` directory.

---

## Evaluation
> ⚠️ **GPU Requirement**: All reported evaluations were conducted on NVIDIA A100 GPUs. We have observed that performance and results may vary across different hardware setups, so results obtained on non-A100 GPUs may differ from those reported here.
### 1. Generate Configs
```bash
python scripts/switch_file_utils.py eval
python scripts/create_mirage_configs.py
```

### 2. Standard and Patch Evals

#### Switch to the original arena
```bash
python scripts/switch_arena.py original
python scripts/run_evals_tmux.py --mode standard --start_index <START> --end_index <END> --gpus <GPUS>
#index corresponds to commands in commands/standard_eval_commands.txt
python scripts/run_evals_tmux.py --mode patch --start_index <START> --end_index <END> --gpus <GPUS>
#index corresponds to commands in commands/patch_eval_commands.txt
```
### 3. Evals with Altered Lighting

#### Switch to the lighting arena
> ⚠️ Do not run this alongside standard or patch evals. Make sure they have completed running first.
```bash
python scripts/switch_arena.py lighting
python scripts/run_evals_tmux.py --mode lighting --start_index <START> --end_index <END> --gpus <GPUS>
#index corresponds to commands in commands/lighting_eval_commands.txt

```

## Results and Visualization

### 1. Collect results
```bash
python scripts/get_eval_results.py
```

### 2. Plot results
```bash
python scripts/plot_results.py
```

>Your results may differ slightly from those reported in the paper due to nondeterminism introduced by EGL.

---
