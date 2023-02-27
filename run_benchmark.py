import random
import time
from sklearn.linear_model import LogisticRegression
from functools import partial
from sparse_benchmark.benchmark import BinaryClassificationBenchmark
import pandas as pd
from sparse_benchmark.transforms import *

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    learner_class_name = "LogisticRegression"
    learner_args = {
        "max_iter": 3000,
        "penalty": None
    }

    featurizers = {
        "StratTfIdf_bag1000" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 1000,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratChi_bag30" :
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 30,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "chi",
                "ranking_learner_args": learner_args,
            }),
        "TfidfVectorizer" :
            ("BenchmarkTfidfVectorizer", {
                "delimiter": '|',
            }),
        "HashingVectorizer" :
            ("BenchmarkHashingVectorizer", {
                "delimiter": '|',
            }),
        "CountVectorizer" :
            ("BenchmarkCountVectorizer", {
                "delimiter": '|',
            }),
    }
    
    
    all_results_raw = []
    all_results_aggr = []
    all_results_merged = []

    # Define the number of features to test
    n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    random.shuffle(n_features)

    for n_features in n_features:
        print(f"\n--- Running benchmark for n_features={n_features} ---")
        for name, featurizer_base in featurizers.items():
            print("Running benchmark for %s..." % name)
            learner = LogisticRegression(**learner_args)
            print(featurizer_base[0])
            featurizer = eval(featurizer_base[0])(**featurizer_base[1])

            # Run the benchmark
            benchmark = BinaryClassificationBenchmark(
                save_folder = f"C://Temp//Data//",
            )
            result_raw, result_aggr, result_merged = benchmark.run(
                featurizer = featurizer,
                learner = learner,
                sample_rate = 0.15,
                n_trials = 5,
                extra_logging = {
                    "featurizer" : name,
                    "n_features" : n_features,
                },
                seed = 42,
            )

            # Save the results
            all_results_aggr.extend(result_aggr)
            all_results_raw.extend(result_raw)
            all_results_merged.extend(result_merged)

            # Print the total roc_auc score
            for r in result_aggr:
                if r["dataset"] == "total":
                    print(f"   roc_auc mean: {r['roc_auc_mean']}")

            # Convert array of dict to df
            df = pd.DataFrame(all_results_aggr)
            df.to_csv("results_aggr.csv")
            df = pd.DataFrame(all_results_raw)
            df.to_csv("results_raw.csv")
            df = pd.DataFrame(all_results_merged)
            df.to_csv("results_merged.csv")



if __name__ == "__main__":
    # Run the benchmark
    run_feature_size_benchmark()
