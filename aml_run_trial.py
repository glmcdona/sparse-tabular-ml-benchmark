import argparse
import os
import numpy as np
import pandas as pd
import json
from azureml.core import Run
from sparse_benchmark.benchmark import BinaryClassificationBenchmark
from sparse_benchmark.transforms import *
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, help="Input folder that contains the csv files.", required=True)
    parser.add_argument("--in_csvs", action="append", help="Filenames to read the data from.", required=True)
    parser.add_argument("--out_folder", type=str, help="Folder to write the data to.")
    parser.add_argument("--out_json_name", type=str, help="Filename to write the data to in json format.")
    parser.add_argument("--name", type=str, help="Name of group of trials.")
    parser.add_argument("--featurizer_class_name", type=str, help="Name of the featurizer class.")
    parser.add_argument("--featurizer_args", type=str, help="Arguments for the featurizer.")
    parser.add_argument("--learner_class_name", type=str, help="Name of the learner class.")
    parser.add_argument("--learner_args", type=str, help="Arguments for the learner.")
    parser.add_argument("--sample_rate", type=float, help="Sample rate.")
    parser.add_argument("--extra_logging", type=str, help="Extra logging.")
    parser.add_argument("--seed", type=int, help="Random seed to start from.")
    args = parser.parse_args()

    # Print arguments
    print(f"Arguments: {args}")

    # Get the Azure ML run
    run = Run.get_context()

    results = []
    for n, in_csv in enumerate(args.in_csvs):
        # Parse the dictionary arguments
        featurizer_args = eval(args.featurizer_args)
        learner_args = eval(args.learner_args)
        extra_logging = eval(args.extra_logging)

        # Load the data
        print(f"Loading data from {os.path.join(args.in_folder, in_csv)}..")
        df = pd.read_csv(os.path.join(args.in_folder, in_csv), dtype={"features": np.str_})
        
        # LR
        if args.learner_class_name == "LogisticRegression":
            learner = LogisticRegression(**learner_args)
        else:
            raise Exception(f"Learner {args.learner_class_name} not found")

        # Create the featurizer
        featurizer = eval(args.featurizer_class_name)(**featurizer_args)

        # Run the benchmark
        benchmark = BinaryClassificationBenchmark()

        result = {"name": args.name}
        result.update(extra_logging)
        
        print(f"Running benchmark for {args.name}..")
        result.update(
            benchmark._run_single(
                featurizer = featurizer,
                learner = learner,
                df = df,
                sample_rate = args.sample_rate,
                extra_logging = extra_logging,
                seed = args.seed + n
            )
        )

        results.append(result)

    # Print the results
    print(results)

    # Save the results
    full_path = os.path.join(args.out_folder, args.out_json_name)
    print(f"Saving results to {full_path}..")
    with open(full_path, "w") as f:
        json.dump(results, f)
    