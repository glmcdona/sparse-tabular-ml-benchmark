import time
from sklearn.linear_model import LogisticRegression
from transforms import BenchmarkBloomVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer
from functools import partial
from benchmark import BinaryClassificationBenchmark
import pandas as pd

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    n_bags = 6
    featurizers = {
        "BloomVectorizer_e0.01_bag6_chi" : partial(
            BenchmarkBloomVectorizer,
            n_bags = n_bags,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "chi",
        ),
        "BloomVectorizer_e0.01_bag6_cv" : partial(
            BenchmarkBloomVectorizer,
            n_bags = n_bags,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "CountVectorizer",
        ),
        "BloomVectorizer_e0.01_bag6_tfidf" : partial(
            BenchmarkBloomVectorizer,
            n_bags = n_bags,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
        ),
        "CountVectorizer" : partial(
            BenchmarkCountVectorizer,
            delimiter = '|',
        ),
        "TfidfVectorizer" : partial(
            BenchmarkTfidfVectorizer,
            delimiter = '|',
        ),
        "HashingVectorizer" : partial(
            BenchmarkHashingVectorizer,
            delimiter = '|',
        ),
    }

    """
        "BloomVectorizer_e0.001_bag6" : partial(
            BenchmarkBloomVectorizer,
            n_bags = 6,
            error_rate = 0.001,
            delimiter = '|',
            task = 'classification',
        ),
        "BloomVectorizer_e0.01_bag6" : partial(
            BenchmarkBloomVectorizer,
            n_bags = 6,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
        ),
        "BloomVectorizer_e0.1_bag6" : partial(
            BenchmarkBloomVectorizer,
            n_bags = 6,
            error_rate = 0.1,
            delimiter = '|',
            task = 'classification',
        ),
    """
    
    all_results_raw = []
    all_results_aggr = []
    #for n_features in [100000, 10000, 1000, 100]:
    for n_features in [100000, 10000]:
        print(f"\n--- Running benchmark for n_features={n_features} ---")
        for name, featurizer_base in featurizers.items():
            print("Running benchmark for %s..." % name)
            featurizer = featurizer_base(
                n_features = n_features,
            )

            learner = LogisticRegression(max_iter=1000)

            # Run the benchmark
            benchmark = BinaryClassificationBenchmark()
            result_raw, result_aggr = benchmark.run(
                featurizer = featurizer,
                learner = learner,
                sample_rate = 1.0,
                n_trials = 1,
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
