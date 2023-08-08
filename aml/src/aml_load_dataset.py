import argparse
import os
import numpy as np
import pandas as pd
from sparse_benchmark.benchmark import standard_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to load.")
    parser.add_argument("--output_file", type=str, help="Filename to write the data to.")
    parser.add_argument("--seed", type=int, help="Random seed to use.")
    args = parser.parse_args()

    # Print arguments
    print(f"Arguments: {args}")

    # Uses the data loader to load the data, and write it as a csv file
    # to the output_data_path

    # Get the dataset
    dataset = args.dataset
    
    # Find the loader
    print(f"Looking for loader for {dataset}...")
    loader = None
    for name, l in standard_datasets:
        if name == dataset:
            loader = l
            break
    
    if loader is None:
        raise Exception(f"Dataset {dataset} not found")
    
    # Load the data
    print(f"Loading data for {dataset} with seed {args.seed}...")
    df = loader(save_folder=None, seed=args.seed)

    # Write the data
    print(f"Writing data to {args.output_file}..")
    df.to_csv(args.output_file, index=False)




    
