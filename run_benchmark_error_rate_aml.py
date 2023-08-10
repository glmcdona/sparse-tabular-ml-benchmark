import numpy as np
from run_aml_benchmark import run_benchmark

if __name__ == "__main__":
    error_rates = np.arange(0, 0.01, 0.0005)
    error_rates = np.append(error_rates, np.arange(0.01, 0.1, 0.005))
    error_rates = np.append(error_rates, np.arange(0.1, 0.55, 0.05))

    run_benchmark(
        "error_rate_benchmark",
        n_bags = [10, 500, 1000, 2000, 3000, 4000, 5000],
        n_features = [100000],
        error_rates = error_rates,
        sample_rates = [1.0],
        stratify_learners = [("tfidf-learner", {"max_iter": 3000, "penalty": None})],
        final_learners = [("LogisticRegression", {"max_iter": 3000, "penalty": None})],
    )
