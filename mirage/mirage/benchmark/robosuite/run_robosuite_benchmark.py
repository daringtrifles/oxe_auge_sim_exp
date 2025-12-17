from mirage.benchmark.robosuite.robosuite_experiment_config import ExperimentRobotsuiteConfig
from mirage.benchmark.robosuite.robosuite_experiment import RobosuiteExperiment

import argparse
import random
import numpy as np
import torch
import os

def setup_deterministic_behavior(seed):
    """
    Set up deterministic behavior for reproducible experiments.
    This ensures all random number generators are seeded properly.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For multi-GPU setups
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Enable deterministic algorithms in PyTorch (available in newer versions)
    try:
        torch.use_deterministic_algorithms(True)
        # Set environment variable for CUDA deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    except AttributeError:
        # Fallback for older PyTorch versions
        torch.backends.cudnn.enabled = False
    
    # Set number of threads to ensure consistent behavior
    torch.set_num_threads(2)
    
    print(f"✓ Deterministic behavior configured with seed: {seed}")

def main():
    parser = argparse.ArgumentParser(description="Mirage Robosuite Benchmark")
    parser.add_argument("--config", type=str)
    parser.add_argument("-y", action="store_true")
    args = parser.parse_args()

    print("Loading config from: ", args.config)
    config = ExperimentRobotsuiteConfig.from_yaml(args.config)
    print(config)
    
    # Set up deterministic behavior using the seed from config
    setup_deterministic_behavior(config.seed)
    
    should_launch = "y"# if args.y else input("Launch the experiment? [Y/n] ")
    if should_launch.lower() != "y":
        print("Exiting...")
        return

    new_experiment = RobosuiteExperiment(config)
    
    try:
        new_experiment.launch()
    except ValueError as e:
        should_override = "y"# if args.y else input("Results folder already exists. Override? [Y/n] ")
        if should_override.lower() != "y":
            print("Exiting...")
            return
        new_experiment.launch(override=True)

    source_results, target_results = new_experiment.get_results(blocking=True)
    print("Source Results:")
    print(source_results)
    print("Target Results:")
    print(target_results)

if __name__ == "__main__":
    main()