import time
from sklearn.linear_model import LogisticRegression
from pipelines import BenchmarkBloomVectorizer
from functools import partial

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    pipeline_classes = {
        "BloomVectorizer_e0.01_bag6" : partial(
            BenchmarkBloomVectorizer,
            n_bags = 6,
            error_rate = 0.01,
        ),
    }
    
    for name, pipeline_class in pipeline_classes.items():
        print("\n--- Running benchmark for %s" % name)
        time_start = time.time()

        benchmark = pipeline_class(
            task = 'classification',
            delimiter = ',',
            n_features = 1000,
            n_bags = 10,
            error_rate = 0.01,
        )






        learner = LogisticRegression(iter=1000)

        


    

    # Run the benchmark
    benchmark.run_benchmark()



if __name__ == "__main__":
    # Run the benchmark
    run_benchmark()
