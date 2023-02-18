import time
from sklearn.linear_model import LogisticRegression
from pipelines import BenchmarkBloomVectorizer
from functools import partial
from benchmark import BinaryClassificationBenchmark
import pandas as pd

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    featurizers = {
        "BloomVectorizer_e0.01_bag6" : partial(
            BenchmarkBloomVectorizer,
            n_bags = 6,
            error_rate = 0.01,
            delimiter = '|',
            task = 'classification',
        ),
    }
    
    all_results = {}
    for n_features in [100, 1000, 10000, 100000]:
        for name, featurizer_base in featurizers.items():
            print("\n--- Running benchmark for %s" % name)
            featurizer = featurizer_base(
                n_features = n_features,
            )

            learner = LogisticRegression(max_iter=1000)

            # Run the benchmark
            benchmark = BinaryClassificationBenchmark()
            results = benchmark.run(
                featurizer = featurizer,
                learner = learner,
                sample_rate = 1.0,
                n_trials = 2,
            )

            # Print the results
            print(results)

            # Save the results
            all_results[name] = results

            # Convert to df
            df = pd.DataFrame.from_dict(all_results)
            df.to_csv("results.csv")



if __name__ == "__main__":
    # Run the benchmark
    run_feature_size_benchmark()
