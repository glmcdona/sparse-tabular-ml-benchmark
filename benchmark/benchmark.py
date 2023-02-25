import time
import copy
from .loaders import loader_newsgroup_binary, loader_click_prediction, loader_airlines, loader_safe_driver, loader_census_income, loader_network_attack, load_bitcoin_ransomware
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from functools import partial

def compute_dataset_properties(df):
    """
    Computes the dataset properties for the given dataframe.
    """
    description = {}

    # Count the distinct number of features
    features = set()
    for f in df["features"].str.split("|"):
        features.update(f)
    description["distinct_features"] = len(features)
    
    # Copute the target distribution
    count_positive = df["target"].value_counts().to_dict()[1]
    description["target_positive"] = count_positive
    description["target_negative"] = len(df) - count_positive
    description["target_rate_positive"] = count_positive / len(df)
    
    return description

class BinaryClassificationBenchmark():
    def __init__(self, datasets=None):
        """Initializes the dataset.
        
        Args:
            dataset_names (list): A list of dataset names to use. If None, all
                datasets will be used.
        """
        if datasets is None:
            datasets = [
                ("bitcoin_ransomware", load_bitcoin_ransomware),
                ("network_attack", loader_network_attack),
                ("census_income", loader_census_income),
                ("safe_driver", loader_safe_driver),
                ("airlines", loader_airlines),
                ("newsgroups", loader_newsgroup_binary),
                ("click_prediction", loader_click_prediction),
            ]
            
        self.datasets = datasets

    def run_pipeline(self, featurizer, learner, loader, seed=42, sample_rate=1.0, n_trials=10):
        """Runs the pipeline on the given dataset.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            loader (): A function that returns the dataframe when given a seed.
            n_trials (int): Number of trials to run for each dataset.
            
        Returns:
            dict: A dictionary containing the results of the benchmark.
        """
        results = {
            "roc_auc": [],
            "f1": [],
            "accuracy": [],
            "precision": [],
            "recall": [],

            "number_of_features": [],
            "train_shape_before_transform": [],
            "train_shape_after_transform": [],
            "test_shape_before_transform": [],
            "test_shape_after_transform": [],
            "size_in_bytes_featurizer": [],
            "size_in_bytes_learner": [],
            "size_in_bytes_total": [],

            "distinct_features": [],
            "target_positive": [],
            "target_negative": [],
            "target_rate_positive": [],

            "time_total": [],
            "time_train_total": [],
            "time_train_transform_fit": [],
            "time_train_transform": [],
            "time_train_fit": [],
            "time_test_total": [],
            "time_test_transform": [],
            "time_test_score": [],
        }
        
        current_seed = seed
        for i in range(n_trials):
            # Load the data with a different seed each time
            df = loader(seed=current_seed)
            current_seed += 1

            # Compute dataset properties
            dataset_properties = compute_dataset_properties(df)

            # Append dataset properties to results
            for k, v in dataset_properties.items():
                results[k].append(v)

            X_train, X_test, y_train, y_test = train_test_split(
                df["features"], df["target"], test_size=0.2)

            # Sample
            if sample_rate < 1.0:
                n_samples = int(len(X_train) * sample_rate)
                X_train = X_train[:n_samples]
                y_train = y_train[:n_samples]

                n_samples = int(len(X_test) * sample_rate)
                X_test = X_test[:n_samples]
                y_test = y_test[:n_samples]
            
            # Transform fit
            start = time.perf_counter_ns()
            featurizer.fit(X_train, y_train)
            results["time_train_transform_fit"].append(time.perf_counter_ns() - start)

            # Transform
            results["train_shape_before_transform"].append(X_train.shape)
            start = time.perf_counter_ns()
            X_train = featurizer.transform(X_train)
            results["time_train_transform"].append(time.perf_counter_ns() - start)
            results["train_shape_after_transform"].append(X_train.shape)

            # Fit learner
            start = time.perf_counter_ns()
            learner.fit(X_train, y_train)
            results["time_train_fit"].append(time.perf_counter_ns() - start)

            # Score transform
            results["test_shape_before_transform"].append(X_test.shape)
            start = time.perf_counter_ns()
            X_test = featurizer.transform(X_test)
            results["time_test_transform"].append(time.perf_counter_ns() - start)
            results["test_shape_after_transform"].append(X_test.shape)

            start = time.perf_counter_ns()
            y_pred = learner.predict(X_test)
            results["time_test_score"].append(time.perf_counter_ns() - start)

            # Compute metrics
            results["roc_auc"].append(roc_auc_score(y_test, y_pred))
            results["f1"].append(f1_score(y_test, y_pred))
            results["accuracy"].append(accuracy_score(y_test, y_pred))
            results["precision"].append(precision_score(y_test, y_pred))
            results["recall"].append(recall_score(y_test, y_pred))

            # Compute total times
            results["time_train_total"].append(
                results["time_train_transform_fit"][-1] +
                results["time_train_transform"][-1] +
                results["time_train_fit"][-1])
            results["time_test_total"].append(
                results["time_test_transform"][-1] +
                results["time_test_score"][-1])
            results["time_total"].append(
                results["time_train_total"][-1] +
                results["time_test_total"][-1])
            
            # Compute featurizer size
            featurizer_size = featurizer.get_size_in_bytes()
            results["size_in_bytes_featurizer"].append(featurizer_size)

            # Save number of features
            results["number_of_features"].append(featurizer.get_num_features())

            # Compute learner size
            learner_size = 0
            if hasattr(learner, "coef_"):
                learner_size += (len(learner.coef_[0]) + 1) * 4
            else:
                print("WARNING: Could not compute learner size.")
            results["size_in_bytes_learner"].append(learner_size)

            # Compute total size
            results["size_in_bytes_total"].append(
                results["size_in_bytes_featurizer"][-1] +
                results["size_in_bytes_learner"][-1])
        
        return results
    
    def run(self, featurizer, learner, sample_rate=1.0, n_trials=10, extra_logging={}, seed=42):
        """Runs the benchmark on the given pipeline.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            n_trials (int): Number of trials to run for each dataset.
            
        Returns:
            dict: A dictionary containing the results of the benchmark.
        """
        results = []
        results_aggregated = []
        result_total = None

        for name, loader in self.datasets:
            print(f"... dataset {name}...")
            result = extra_logging.copy()
            result["dataset"] = name
            result["n_trials"] = n_trials
            if sample_rate < 1.0:
                result["sample_rate"] = sample_rate

            result.update(
                self.run_pipeline(featurizer, learner,
                    loader, seed, sample_rate, n_trials)
            )
            results.append(copy.deepcopy(result))

            # Add the result to the total dataset-independent results
            if result_total is None:
                result_total = copy.deepcopy(result)
                result_total["dataset"] = "total"
            else:
                for key, value in result.items():
                    if isinstance(value, list) and len(value) == n_trials and "shape" not in key:
                        result_total[key].extend(value)
        
            # Add an aggregated results taking mean and stddev of all trials
            result_aggregated = extra_logging.copy()
            result_aggregated["dataset"] = name
            for key, value in result.items():
                # Check if it's an array
                if isinstance(value, list) and len(value) == n_trials and "shape" not in key:
                    result_aggregated[key + "_mean"] = np.mean(value)
                    result_aggregated[key + "_stddev"] = np.std(value)
                else:
                    result_aggregated[key] = value
            results_aggregated.append(result_aggregated)
        
        # Aggregate the total dataset-independent results
        result_total_agg = {
            "dataset": "total",
        }
        for key, value in result_total.items():
            # Check if it's an array
            if isinstance(value, list) and "shape" not in key:
                result_total_agg[key + "_mean"] = np.mean(value)
                result_total_agg[key + "_stddev"] = np.std(value)
                result_total_agg["n_trials"] = len(value)
            elif key == "n_trials":
                pass
            else:
                result_total_agg[key] = value
        
        results_aggregated.append(result_total_agg)
        
        return results, results_aggregated
