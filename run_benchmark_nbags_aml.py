import numpy as np
from aml_benchmark_runner import run_benchmark

if __name__ == "__main__":
    n_bags = np.arange(1, 10, 1, dtype=int)
    n_bags = np.append(n_bags, np.arange(10, 100, 5))
    n_bags = np.append(n_bags, np.arange(100, 1000, 50))
    n_bags = np.append(n_bags, np.arange(1000, 10000, 500))
    n_bags = np.append(n_bags, np.arange(10000, 100000, 5000))
    n_bags = [int(x) for x in n_bags]

    run_benchmark(
        "nbags_benchmark",
        n_bags = n_bags,
        n_features = [100000],
        error_rates = [0],
        sample_rates = [0.1, 0.5, 1.0],
        stratify_learners = [("tfidf-learner", {"max_iter": 3000, "penalty": None})],
        final_learners = [("LogisticRegression", {"max_iter": 3000, "penalty": None})],
    )
