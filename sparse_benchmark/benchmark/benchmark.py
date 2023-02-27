import json
import os
import shutil
import tempfile
import time
import copy
from .loaders import loader_newsgroup, loader_click_prediction, loader_airlines, loader_safe_driver, loader_census_income, loader_network_attack, loader_bitcoin_ransomware
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from functools import partial


standard_datasets = [
    ("bitcoin_ransomware", loader_bitcoin_ransomware),
    ("network_attack", loader_network_attack),
    ("census_income", loader_census_income),
    ("safe_driver", loader_safe_driver),
    ("airlines", loader_airlines),
    ("newsgroup", loader_newsgroup),
    ("click_prediction", loader_click_prediction),
]

def compute_dataset_properties(df):
    """
    Computes the dataset properties for the given dataframe.
    """
    description = {}

    # Count the distinct number of features
    features = set()
    for f in df["features"].str.split("|"):
        try:
            features.update(f)
        except:
            print("ERROR", f)

    description["distinct_features"] = len(features)
    
    # Compute the target distribution
    count_positive = df["target"].value_counts().to_dict()[1]
    description["target_positive"] = count_positive
    description["target_negative"] = len(df) - count_positive
    description["target_rate_positive"] = count_positive / len(df)
    
    return description

class BinaryClassificationBenchmark():
    def __init__(self, datasets=None, save_folder=None):
        """Initializes the dataset.
        
        Args:
            dataset_names (list): A list of dataset names to use. If None, all
                datasets will be used.
        """
        if datasets is None:
            datasets = standard_datasets
            
        self.datasets = datasets

        if save_folder is None:
            self.save_folder = os.path.join(os.getcwd(), "data")
        else:
            self.save_folder = save_folder
        
        self.queue = []

    def _run_single(self, featurizer, learner, df, sample_rate=1.0, extra_logging=None, seed=42):
        """Runs the pipeline on the given dataset.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            df (pandas.DataFrame): The dataset to run the pipeline on.
            sample_rate (float): The sample rate to use for the dataset.
            
        Returns:
            dict: A dictionary containing the results of the benchmark run.
        """
        # copy extra_logging
        if extra_logging is None:
            results = {}
        else:
            results = copy.deepcopy(extra_logging)


        results.update({
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
        })

        # Compute dataset properties
        dataset_properties = compute_dataset_properties(df)

        # Append dataset properties to results
        for k, v in dataset_properties.items():
            results[k].append(v)

        X_train, X_test, y_train, y_test = train_test_split(
            df["features"], df["target"], test_size=0.2, random_state=seed)

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

        # Scale the data
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)

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

        # Scale the data
        X_test = scaler.transform(X_test)

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

    def _run_trials(self, featurizer, learner, loader, seed=42, sample_rate=1.0, n_trials=10, extra_logging={}):
        """Runs the pipeline on the given dataset.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            loader (): A function that returns the dataframe when given a seed.
            n_trials (int): Number of trials to run for each dataset.
            
        Returns:
            list: A list of dictionaries containing the results of the benchmark.
        """
        current_seed = seed
        results = []
        for i in range(n_trials):
            # Load the data with a different seed each time
            df = loader(seed=current_seed, save_folder=self.save_folder)

            result = copy.deepcopy(extra_logging)

            result.update(
                self._run_single(
                                featurizer, learner, df,
                                sample_rate=sample_rate,
                                extra_logging={"trial": i, "seed": current_seed},
                                seed=current_seed
                            )
                        )
            
            results.append(result)

            current_seed += 1
        
        return results
    
    def _aggregate_results(self, results, group_by=["dataset"]):
        """Aggregates the results of the benchmark.
        
        Args:
            results (list): A list of dictionaries containing the results of the benchmark.
            
        Returns:
            dict: A dictionary containing the aggregated results of the benchmark.
        """

        # Aggregate all the results together, calculating the mean and standard
        # deviation of each column if it is a list.


        # First merge all with the same group_by key values
        # Merging is done by appending the values of the lists, and taking the
        # first value of the other columns.
        results_merged = {}
        for result in results:
            key = tuple([result[k] for k in group_by])
            
            if key not in results_merged:
                results_merged[key] = {}
            for k, v in result.items():
                if isinstance(v, list) and "shape" not in k:
                    if k not in results_merged[key]:
                        results_merged[key][k] = []
                    results_merged[key][k].extend(v)
                elif k in ["trial", "seed"]:
                    # Skip these
                    continue
                elif k not in results_merged[key]:
                    results_merged[key][k] = v

        # Then calculate the mean and std of the lists
        results_aggregated = []
        for key, result in results_merged.items():
            result_aggregated = {}
            for k, v in result.items():
                if isinstance(v, list) and "shape" not in k:
                    result_aggregated[k + "_mean"] = np.mean(v)
                    result_aggregated[k + "_stddev"] = np.std(v)
                else:
                    result_aggregated[k] = v
            results_aggregated.append(result_aggregated)
        
        return results_aggregated, results_merged.values()

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

        for name, loader in self.datasets:
            print(f"... dataset {name}...")
            result = copy.deepcopy(extra_logging)
            result["dataset"] = name
            result["n_trials"] = n_trials
            if sample_rate < 1.0:
                result["sample_rate"] = sample_rate

            trial_results = self._run_trials(featurizer, learner,
                    loader, seed, sample_rate, n_trials)

            for trial_result in trial_results:
                trial_result_merged = copy.deepcopy(result)
                trial_result_merged.update(trial_result)
                results.append(trial_result_merged)
            
        group_by = ["dataset"]
        group_by.extend(extra_logging.keys())
        results_aggregated, results_merged = self._aggregate_results(
            results,
            group_by=group_by
        )
        
        return results, results_aggregated, results_merged

    def add_queue(self, name, featurizer_class_name, featurizer_args, learner_class_name, learner_args={}, sample_rate=1.0, extra_logging={}):
        """Queues a run of the benchmark on the given pipeline.
        
        Args:
            featurizer (Featurizer): A featurizer transform.
            learner (Learner): A learner to fit and score.
            n_trials (int): Number of trials to run for each dataset.
        """
        self.queue.append(
            {
                "name": name,
                "featurizer_class_name": featurizer_class_name,
                "featurizer_args": featurizer_args,
                "learner_class_name": learner_class_name,
                "learner_args": learner_args,
                "sample_rate": sample_rate,
                "extra_logging": extra_logging,
            }
        )

    def run_queue_azure_aml(
                self,
                workspace,
                datastore,
                experiment_name,
                compute_target,
                data_path,
                output_path,
                n_trials = 10,
                seed = 42,
            ):
        """
        Runs all queued pipelines on Azure ML. It will first run
        featurization distributed on the cluster, then run the
        learning on the cluster, and finally aggregate the results
        here.
        
        Args:
            workspace (Workspace): The Azure ML workspace.
            datastore (Datastore): The Azure ML datastore.
            experiment_name (str): The name of the experiment.
            compute_target (ComputeTarget): The Azure ML compute target.
            data_path (str): The path to the data.
            output_path (str): The path to the output.
            n_trials (int): Number of trials to run for each dataset.
            seed (int): The random seed.
        """
        import azureml.core
        from azureml.core import Workspace,  Experiment, Dataset
        from azureml.core.compute import AmlCompute
        from azureml.core.runconfig import RunConfiguration
        from azureml.core.conda_dependencies import CondaDependencies
        import azureml
        from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
        from azureml.pipeline.steps import PythonScriptStep
        from azureml.data import FileDataset, OutputFileDatasetConfig
        from azureml.data.datapath import DataPath, DataPathComputeBinding


        print("Using AzureML SDK version:", azureml.core.VERSION)
        

        # Download and create the datasets first, some of them take a while
        # so we do this in parallel
        experiment = Experiment(workspace, experiment_name)
        steps = []
        
        for dataset in self.datasets:
            print(f"Loading {dataset[0]}...")
            
            for trial in range(n_trials):
                name = f"loader_{dataset[0]}_{seed + trial}"

                step_output_path = f"{data_path}/{dataset[0]}/"
                output_name = f"{dataset[0]}_{seed + trial}.csv"

                # Check if the file already exists in the datastore already
                try:
                    file_dataset = Dataset.File.from_files(path=(datastore, os.path.join(step_output_path, output_name)))
                    if file_dataset is not None:
                        print(f"Skipping {name} because data already exists")
                        continue
                except:
                    pass
                
                step_output_path = OutputFileDatasetConfig(
                    name="step_output_path",
                    destination=(datastore, step_output_path)
                )

                # Create the step
                steps.append(
                    PythonScriptStep(
                        name=name,
                        script_name="aml_load_dataset.py",
                        arguments=[
                            "--dataset", dataset[0],
                            "--output_path", step_output_path,
                            "--output_name", output_name,
                            "--seed", seed + trial,
                        ],
                        inputs=[],
                        outputs=[step_output_path],
                        compute_target=compute_target,
                        source_directory=".",
                        runconfig=RunConfiguration(
                                conda_dependencies=CondaDependencies.create(
                                    conda_packages=[], 
                                    pip_packages=['azureml-sdk', 'numpy', 'pandas', 'scikit-learn'], 
                                    pin_sdk_version=False)
                            ),
                    )
                )

        if len(steps) > 0:
            # Create the pipeline
            pipeline = Pipeline(workspace=workspace, steps=steps)
            pipeline_run = experiment.submit(pipeline, continue_on_step_failure=True)
            
            # Wait for the data preparation pipeline to finish
            pipeline_run.wait_for_completion()
        
        # Now run the featurization and learning jobs
        print("Creating benchmark trial runs...")
        steps = []
        for id, queue_item in enumerate(self.queue):
            for dataset in self.datasets:
                for trial in range(n_trials):
                    name = f"{queue_item['name']}_{dataset[0]}_{seed + trial}_id{id}"

                    output_name = f"{name}.json"

                    step_output_path = OutputFileDatasetConfig(
                        name="output_path",
                        destination=(datastore, f"{output_path}")
                    )

                    # Create the input folder
                    input_folder = f"{data_path}/{dataset[0]}/"
                    input_name = f"{dataset[0]}_{seed + trial}.csv"

                    input_file_path = DataPath(datastore, os.path.join(input_folder, input_name))
                    data_path_pipeline_param = (PipelineParameter(name=name, default_value=input_file_path),
                                                DataPathComputeBinding(mode='mount'))
                    
                    # Create the step
                    steps.append(
                        PythonScriptStep(
                            name=name,
                            script_name="aml_run_trial.py",
                            arguments=[
                                "--in_csv", data_path_pipeline_param,
                                "--out_folder", step_output_path,
                                "--out_json_name", output_name,
                                "--name", name,
                                "--featurizer_class_name", queue_item['featurizer_class_name'],
                                "--featurizer_args", str(queue_item['featurizer_args']),
                                "--learner_class_name", queue_item['learner_class_name'],
                                "--learner_args", str(queue_item['learner_args']),
                                "--sample_rate", queue_item['sample_rate'],
                                "--extra_logging", str(queue_item['extra_logging']),
                                "--seed", seed + trial,
                            ],
                            inputs=[data_path_pipeline_param],
                            outputs=[step_output_path],
                            compute_target=compute_target,
                            source_directory=".",
                            runconfig=RunConfiguration(
                                    conda_dependencies=CondaDependencies.create(
                                        conda_packages=[], 
                                        pip_packages=['azureml-sdk', 'numpy', 'pandas', 'scikit-learn',
                                            "git+https://github.com/glmcdona/stratified-vectorizer.git"],
                                        pin_sdk_version=False
                                    )
                                ),
                        )
                    )
        
        # Create the pipeline
        print("Submitting trial benchmark runs...")
        pipeline = Pipeline(workspace=workspace, steps=steps)
        pipeline_run = experiment.submit(pipeline, continue_on_step_failure=True)

        # Wait for the data preparation pipeline to finish
        pipeline_run.wait_for_completion()

        # Now merge the results
        print("Merging results...")

        # Check if the file already exists in the datastore already
        results = []
        for id, queue_item in enumerate(self.queue):
            for dataset in self.datasets:
                for trial in range(n_trials):
                    name = f"{queue_item['name']}_{dataset[0]}_{seed + trial}_id{id}"
                    output_name = f"{name}.json"
                    try:
                        file_dataset = Dataset.File.from_files(path=(datastore, os.path.join(output_path, output_name)))
                        if file_dataset is not None:
                            # Open the file and read the results
                            print(f"Found {name} in the datastore. Adding to results.")
                            files = file_dataset.download()
                            print(files)

                            with open(files[0], 'r') as f:
                                # Print the file contents
                                data = {"dataset": dataset[0]}
                                data.update(json.load(f))
                                results.append(data)
                    except:
                        print(f"ERROR: Could not find {name} in the datastore. Skipping this result.")
                        pass
        
        # Aggregate the results
        group_by = ["dataset"]
        if len(self.queue) > 1:
            group_by.extend( self.queue[0]["extra_logging"].keys() )
        
        results_aggregated, results_merged = self._aggregate_results(results, group_by=group_by)

        return results, results_aggregated, results_merged

        



        
                





            


            
        