import random
import time
from sklearn.linear_model import LogisticRegression
from transforms import BenchmarkStratifiedBagVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer, BenchmarkMultiVectorizer
from functools import partial
from benchmark import BinaryClassificationBenchmark
import pandas as pd

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    base_learner = partial(
        LogisticRegression,
        max_iter=3000,
        penalty=None
    )
    featurizers = {
        "TfidfVectorizer" : partial(
            BenchmarkTfidfVectorizer,
            delimiter = '|',
        ),
        "HashingVectorizer" : partial(
            BenchmarkHashingVectorizer,
            delimiter = '|',
        ),
        "CountVectorizer" : partial(
            BenchmarkCountVectorizer,
            delimiter = '|',
        ),
        "StratTfIdf_bag1000" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 1000,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
            ranking_learner = base_learner()
        ),
        "StratTfIdf_bag300" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 300,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
            ranking_learner = base_learner()
        ),
        "StratTfIdf_bag30" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 30,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
            ranking_learner = base_learner()
        ),
        "StratChi_bag30" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 30,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "chi",
            ranking_learner = base_learner()
        ),
    }
    
    
    all_results_raw = []
    all_results_aggr = []

    # Define the number of features to test
    n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    random.shuffle(n_features)

    for n_features in n_features:
        print(f"\n--- Running benchmark for n_features={n_features} ---")
        for name, featurizer_base in featurizers.items():
            print("Running benchmark for %s..." % name)
            featurizer = featurizer_base(
                n_features = n_features,
            )

            # LR with no regularization
            learner = base_learner()

            # Run the benchmark
            benchmark = BinaryClassificationBenchmark()
            result_raw, result_aggr = benchmark.run(
                featurizer = featurizer,
                learner = learner,
                sample_rate = 0.05,
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

            # Print the total roc_auc score
            for r in result_aggr:
                if r["dataset"] == "total":
                    print(f"   roc_auc mean: {r['roc_auc_mean']}")

            # Convert array of dict to df
            df = pd.DataFrame(all_results_aggr)
            df.to_csv("results_aggr.csv")
            df = pd.DataFrame(all_results_raw)
            df.to_csv("results_raw.csv")



if __name__ == "__main__":
    # Run the benchmark
    run_feature_size_benchmark()
