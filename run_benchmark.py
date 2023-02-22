import random
import time
from sklearn.linear_model import LogisticRegression
from transforms import BenchmarkStratifiedBagVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer
from functools import partial
from benchmark import BinaryClassificationBenchmark
import pandas as pd

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    featurizers = {
        "StratifiedBagVectorizer_e0_bag30_chi" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 30,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "chi",
        ),
        "StratifiedBagVectorizer_e0_bag300_chi" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 300,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "chi",
        ),
        "StratifiedBagVectorizer_e0_bag30_tfidf" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 30,
            error_rate = 0,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
        ),
        "StratifiedBagVectorizer_e0_bag300_tfidf" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 300,
            error_rate = 0,
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
    "StratifiedBagVectorizer_e0.01_bag6_chi" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 6,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "chi",
        ),
        "StratifiedBagVectorizer_e0.01_bag6_cv" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 6,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "CountVectorizer",
        ),
        "StratifiedBagVectorizer_e0.01_bag6_tfidf" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 6,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
        ),
        "StratifiedBagVectorizer_e0.001_bag15_tfidf" : partial(
            BenchmarkStratifiedBagVectorizer,
            n_bags = 15,
            error_rate = 0.001,
            delimiter = '|',
            task = 'classification',
            ranking_method = "TfidfVectorizer",
        ),

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
    #for n_features in [100000, 10000]:
    while(True):
        rand = random.random()
        if rand < 0.25:
            n_features = random.randint(10, 1000)
        elif rand < 0.50:
            n_features = random.randint(1000, 10000)
        elif rand < 0.75:
            n_features = random.randint(10000, 100000)
        else:
            n_features = random.randint(100000, 500000)

        print(f"\n--- Running benchmark for n_features={n_features} ---")
        for name, featurizer_base in featurizers.items():
            print("Running benchmark for %s..." % name)
            featurizer = featurizer_base(
                n_features = n_features,
            )

            # LR with no regularization
            learner = LogisticRegression(max_iter=3000, penalty=None)

            # Run the benchmark
            benchmark = BinaryClassificationBenchmark()
            result_raw, result_aggr = benchmark.run(
                featurizer = featurizer,
                learner = learner,
                sample_rate = 0.25,
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
