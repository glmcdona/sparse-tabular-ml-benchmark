import numpy as np
from aml_benchmark_runner import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        "lightgbm_benchmark",
        #n_bags = [10, 100, 1000, 5000, 10000, 100000],
        n_bags = [5000, 10000, 100000],
        #n_features = [5000, 100000, 500000],
        n_features = [500000],
        error_rates = [0],
        #sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
        sample_rates = [0.1],
        stratify_learners = [("tfidf-learner", {"max_iter": 3000, "penalty": None})],
        final_learners = [
            ("LogisticRegression", {"max_iter": 3000, "penalty": None}),
            ("LGBMClassifier", {})
        ],
    )
