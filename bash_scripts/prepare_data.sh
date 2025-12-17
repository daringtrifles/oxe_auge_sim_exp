#!/bin/bash

# Check if input data path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_data_path>"
    echo "Example: $0 /path/to/your/input/data"
    exit 1
fi

INPUT_DATA_PATH="$1"

# Run split_hdf5.py with input data path
python scripts/split_hdf5.py --input_data_path "$INPUT_DATA_PATH"

python scripts/merge_data.py
