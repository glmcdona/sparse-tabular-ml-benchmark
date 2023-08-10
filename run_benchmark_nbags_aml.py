import numpy as np
from run_aml_benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        "nbags_benchmark",
        n_bags = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000],
        n_features = [100000],
        error_rates = [0],
        sample_rates = [0.1, 0.5, 1.0],
        stratify_learners = [("tfidf-learner", {"max_iter": 3000, "penalty": None})],
        final_learners = [("LogisticRegression", {"max_iter": 3000, "penalty": None})],
    )
