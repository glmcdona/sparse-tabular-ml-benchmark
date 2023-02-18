import time
from .loaders import loader_newsgroup_binary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BinaryClassificationBenchmark():
    def __init__(self, datasets=None):
        """Initializes the dataset.
        
        Args:
            dataset_names (list): A list of dataset names to use. If None, all
                datasets will be used.
        """
        if datasets is None:
            datasets = {
                "newsgroups": loader_newsgroup_binary
            }
            
        self.datasets = datasets

    def get_next_benchmark(self):
        """Generator returns the next benchmark to run.
        
        Returns:
            tuple: A tuple containing the name of the benchmark, dataframe
                containing "features" and "target" columns.
        """
        for name, loader in self.datasets.items():
            yield name, loader()

    def run_pipeline(self, featurizer, learner, df, sample_rate=1.0, n_trials=10):
        """Runs the pipeline on the given dataset.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            df (DataFrame): A dataframe containing the "features" and "target"
                columns.
            n_trials (int): Number of trials to run for each dataset.
            
        Returns:
            dict: A dictionary containing the results of the benchmark.
        """
        results = {
            "time": {
                "total": [],
                "train_total": [],
                "train_transform_fit": [],
                "train_transform": [],
                "train_fit": [],
                "test_total": [],
                "test_transform": [],
                "test_score": [],
            },
            "size": {
                "number_of_features": [],
                "train_shape_before_transform": [],
                "train_shape_after_transform": [],
                "test_shape_before_transform": [],
                "test_shape_after_transform": [],
                "size_in_bytes_featurizer": [],
                "size_in_bytes_learner": [],
                "size_in_bytes_total": [],
            },
            "score": {
                "roc_auc": [],
                "f1": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
            },
        }
        
        for i in range(n_trials):
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
            start = time.time()
            featurizer.fit(X_train, y_train)
            results["time"]["train_transform_fit"].append(time.time() - start)

            # Transform
            results["size"]["train_shape_before_transform"].append(X_train.shape)
            start = time.time()
            X_train = featurizer.transform(X_train)
            results["time"]["train_transform"].append(time.time() - start)
            results["size"]["train_shape_after_transform"].append(X_train.shape)

            # Fit learner
            start = time.time()
            learner.fit(X_train, y_train)
            results["time"]["train_fit"].append(time.time() - start)

            # Score transform
            results["size"]["test_shape_before_transform"].append(X_test.shape)
            start = time.time()
            X_test = featurizer.transform(X_test)
            results["time"]["test_transform"].append(time.time() - start)
            results["size"]["test_shape_after_transform"].append(X_test.shape)

            start = time.time()
            y_pred = learner.predict(X_test)
            results["time"]["test_score"].append(time.time() - start)

            # Compute metrics
            results["score"]["roc_auc"].append(roc_auc_score(y_test, y_pred))
            results["score"]["f1"].append(f1_score(y_test, y_pred))
            results["score"]["accuracy"].append(accuracy_score(y_test, y_pred))
            results["score"]["precision"].append(precision_score(y_test, y_pred))
            results["score"]["recall"].append(recall_score(y_test, y_pred))

            # Compute total times
            results["time"]["train_total"].append(
                results["time"]["train_transform_fit"][-1] +
                results["time"]["train_transform"][-1] +
                results["time"]["train_fit"][-1])
            results["time"]["test_total"].append(
                results["time"]["test_transform"][-1] +
                results["time"]["test_score"][-1])
            results["time"]["total"].append(
                results["time"]["train_total"][-1] +
                results["time"]["test_total"][-1])
            
            # Compute featurizer size
            featurizer_size = featurizer.get_size_in_bytes()
            results["size"]["size_in_bytes_featurizer"].append(featurizer_size)

            # Save number of features
            results["size"]["number_of_features"].append(featurizer.get_num_features())

            # Compute learner size
            learner_size = 0
            if hasattr(learner, "coef_"):
                learner_size += learner.coef_[0] * 4 + 1
            else:
                print("WARNING: Could not compute learner size.")
            results["size"]["size_in_bytes_learner"].append(learner_size)

            # Compute total size
            results["size"]["size_in_bytes_total"].append(
                results["size"]["size_in_bytes_featurizer"][-1] +
                results["size"]["size_in_bytes_learner"][-1])
            
        return results
    
    def run(self, featurizer, learner, sample_rate=1.0, n_trials=10):
        """Runs the benchmark on the given pipeline.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            n_trials (int): Number of trials to run for each dataset.
            
        Returns:
            dict: A dictionary containing the results of the benchmark.
        """
        results = {}
        for name, df in self.get_next_benchmark():
            results[name] = self.run_pipeline(featurizer, learner,
                df, sample_rate, n_trials)
            
        return results
