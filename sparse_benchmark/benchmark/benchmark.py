import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import copy
from .loaders import loader_newsgroup, loader_click_prediction, loader_airlines, loader_safe_driver, loader_census_income, loader_network_attack, loader_bitcoin_ransomware
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from functools import partial

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azureml.fsspec import AzureMachineLearningFileSystem
from azure.ai.ml import command, Input, Output, MLClient, UserIdentityConfiguration, ManagedIdentityConfiguration
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential


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
            "test_roc_auc": [],
            "test_f1": [],
            "test_accuracy": [],
            "test_precision": [],
            "test_recall": [],

            "valid_roc_auc": [],
            "valid_f1": [],
            "valid_accuracy": [],
            "valid_precision": [],
            "valid_recall": [],

            "test_and_valid_roc_auc": [],
            "test_and_valid_f1": [],
            "test_and_valid_accuracy": [],
            "test_and_valid_precision": [],
            "test_and_valid_recall": [],

            "train_roc_auc": [],
            "train_f1": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],

            "number_of_features": [],
            "train_shape_before_transform": [],
            "train_shape_after_transform": [],
            "test_shape_before_transform": [],
            "test_shape_after_transform": [],
            "valid_shape_before_transform": [],
            "valid_shape_after_transform": [],
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
            "time_train_score": [],
            "time_test_total": [],
            "time_test_transform": [],
            "time_test_score": [],
            "time_valid_total": [],
            "time_valid_transform": [],
            "time_valid_score": [],
        })

        # Compute dataset properties
        dataset_properties = compute_dataset_properties(df)

        # Append dataset properties to results
        for k, v in dataset_properties.items():
            results[k].append(v)

        # Split into train, test, and validation
        # 70% train, 15% test, 15% validation
        X_train, X_hold, y_train, y_hold = train_test_split(
            df["features"], df["target"], test_size=0.3, random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(
            X_hold, y_hold, test_size=0.5, random_state=seed)

        # Sample
        if sample_rate < 1.0:
            n_samples = int(len(X_train) * sample_rate)
            X_train = X_train[:n_samples]
            y_train = y_train[:n_samples]

            n_samples = int(len(X_test) * sample_rate)
            X_test = X_test[:n_samples]
            y_test = y_test[:n_samples]

            n_samples = int(len(X_val) * sample_rate)
            X_val = X_val[:n_samples]
            y_val = y_val[:n_samples]
        
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

        # Score and compute the train metrics
        start = time.perf_counter_ns()
        y_pred = learner.predict(X_train)
        results["time_train_score"].append(time.perf_counter_ns() - start)

        results["train_roc_auc"].append(roc_auc_score(y_train, y_pred))
        results["train_f1"].append(f1_score(y_train, y_pred))
        results["train_accuracy"].append(accuracy_score(y_train, y_pred))
        results["train_precision"].append(precision_score(y_train, y_pred))
        results["train_recall"].append(recall_score(y_train, y_pred))

        # Transform test and validation data
        results["test_shape_before_transform"].append(X_test.shape)
        start = time.perf_counter_ns()
        X_test = featurizer.transform(X_test)
        results["time_test_transform"].append(time.perf_counter_ns() - start)
        results["test_shape_after_transform"].append(X_test.shape)

        results["valid_shape_before_transform"].append(X_val.shape)
        start = time.perf_counter_ns()
        X_val = featurizer.transform(X_val)
        results["time_valid_transform"].append(time.perf_counter_ns() - start)
        results["valid_shape_after_transform"].append(X_val.shape)

        # Scale the data
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        # Score and compute the test and validation metrics
        start = time.perf_counter_ns()
        y_pred_test = learner.predict(X_test)
        results["time_test_score"].append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        y_pred_val = learner.predict(X_val)
        results["time_valid_score"].append(time.perf_counter_ns() - start)
        
        results["test_roc_auc"].append(roc_auc_score(y_test, y_pred_test))
        results["test_f1"].append(f1_score(y_test, y_pred_test))
        results["test_accuracy"].append(accuracy_score(y_test, y_pred_test))
        results["test_precision"].append(precision_score(y_test, y_pred_test))
        results["test_recall"].append(recall_score(y_test, y_pred_test))

        results["valid_roc_auc"].append(roc_auc_score(y_val, y_pred_val))
        results["valid_f1"].append(f1_score(y_val, y_pred_val))
        results["valid_accuracy"].append(accuracy_score(y_val, y_pred_val))
        results["valid_precision"].append(precision_score(y_val, y_pred_val))
        results["valid_recall"].append(recall_score(y_val, y_pred_val))

        results["test_and_valid_roc_auc"].append(roc_auc_score(
            np.concatenate([y_test, y_val]),
            np.concatenate([y_pred_test, y_pred_val]))
        )
        results["test_and_valid_f1"].append(f1_score(
            np.concatenate([y_test, y_val]),
            np.concatenate([y_pred_test, y_pred_val]))
        )
        results["test_and_valid_accuracy"].append(accuracy_score(
            np.concatenate([y_test, y_val]),
            np.concatenate([y_pred_test, y_pred_val]))
        )
        results["test_and_valid_precision"].append(precision_score(
            np.concatenate([y_test, y_val]),
            np.concatenate([y_pred_test, y_pred_val]))
        )
        results["test_and_valid_recall"].append(recall_score(
            np.concatenate([y_test, y_val]),
            np.concatenate([y_pred_test, y_pred_val]))
        )

        # Compute total times
        results["time_train_total"].append(
            results["time_train_transform_fit"][-1] +
            results["time_train_transform"][-1] +
            results["time_train_fit"][-1])
        results["time_test_total"].append(
            results["time_test_transform"][-1] +
            results["time_test_score"][-1])
        results["time_valid_total"].append(
            results["time_valid_transform"][-1] +
            results["time_valid_score"][-1])
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
                ml_client : MLClient,
                compute_target,
                root_storage_path,
                relative_data_path,
                experiment_name,
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

        # Create the environment if it doesn't exist yet
        custom_env_name = "sparse-featurizer-env"
        dependencies_dir = "./aml/env"
        source_dir = "./aml/src"

        job_envs = list(ml_client.environments.list(custom_env_name))
        if len(job_envs) == 0 or True:
            # Create the environment
            job_env = Environment(
                name=custom_env_name,
                description="Custom environment for sparse featurizer experiments",
                conda_file=os.path.join(dependencies_dir, "conda.yaml"),
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            )

            job_env = ml_client.environments.create_or_update(job_env)

            print(
                f"Environment with name {job_env.name} is registered to workspace, the environment version is {job_env.version}"
            )
        else:
            job_env = job_envs[-1]
            print(f"Using existing environment with name {job_env.name} and version {job_env.version}")
        
        # Copy sparse_benchmark into the dependencies dir for AML to use
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        shutil.rmtree(os.path.join(source_dir, "sparse_benchmark"), ignore_errors=True)
        shutil.copytree("./sparse_benchmark", os.path.join(source_dir, "sparse_benchmark"))
        
        fs = AzureMachineLearningFileSystem(root_storage_path)

        # Download and create the datasets first, some of them take a while
        jobs = []
        for dataset in self.datasets:
            print(f"Loading {dataset[0]}...")

            # Load the csv files for this dataset
            dataset_csvs = fs.glob(path=f"{relative_data_path}/data/{dataset[0]}/*.csv")
            print(f"Found {len(dataset_csvs)} datasets in the datastore for {dataset[0]}")
            print(dataset_csvs)
            
            for trial in range(n_trials):
                name = f"loader_{dataset[0]}_{seed + trial}"

                output_name = f"{dataset[0]}_{seed + trial}.csv"
                output_full_path = f"azureml://datastores/workspaceblobstore/paths/{relative_data_path}/data/{dataset[0]}/{output_name}"

                # Check if the file already exists in the datastore
                #if f"{relative_data_path}/data/{dataset[0]}/{output_name}" in dataset_csvs:
                print(os.path.join(relative_data_path, "data", dataset[0], output_name))
                if os.path.join(relative_data_path, "data", dataset[0], output_name).replace("\\", "/") in dataset_csvs:
                    print(f"Skipping {name} because data already exists")
                    continue
                
                print(f"Creating step {name} for {dataset[0]}")

                # Create the job
                job = command(
                    inputs=dict(
                        dataset = dataset[0],
                        seed = seed + trial,
                    ),
                    outputs = dict(
                        output_file = Output(
                            type = AssetTypes.URI_FILE, 
                            path = output_full_path, 
                            mode = InputOutputModes.RW_MOUNT,
                        )
                    ),
                    compute = compute_target,
                    environment = f"{job_env.name}:{job_env.version}",
                    code = "./aml/src/",
                    command = "python aml_load_dataset.py --dataset ${{inputs.dataset}} --output_file ${{outputs.output_file}} --seed ${{inputs.seed}}",
                    experiment_name = f"sparse-featurizer-data-prep",
                    display_name = name,
                )

                # Submit the command
                jobs.append(ml_client.jobs.create_or_update(job))

        if len(jobs) > 0:
            # Wait for completion
            print("Waiting for data loading jobs to complete...")
            
            while True:
                status = {}
                for job in jobs:
                    job_status = ml_client.jobs.get(name=job.name).status
                    if job_status not in status:
                        status[job_status] = 0
                    status[job_status] += 1
                
                print(status)

                if all([s in ["Completed", "Failed"] for s in status.keys()]):
                    break
                
                time.sleep(5)
            
            # Check for failures
            if any([s == "Failed" for s in status.keys()]):
                raise RuntimeError("One or more data loading jobs failed")
            
            print("Data loading jobs completed successfully")
                

        
        # Now run the featurization and learning jobs
        print("Creating benchmark trial runs...")

        # Load existing results
        dataset_jsons = fs.glob(path=f"{relative_data_path}/results/*.json")
        print(f"Found {len(dataset_csvs)} json file results")
        print(dataset_jsons)
        jobs = []
        for queue_item in self.queue:
            m = hashlib.sha256()
            m.update(json.dumps(queue_item).encode("utf-8"))
            m.update(str(n_trials).encode("utf-8"))
            m.update(str(seed).encode("utf-8"))
            id = m.hexdigest()[:8]

            for dataset in self.datasets:
                name = f"{queue_item['name']}_{dataset[0]}_{id}"

                # Create the output json file
                output_name = f"{name}.json"

                # Check if the file already exists in the datastore already
                print(f"{relative_data_path}/results/{output_name}")
                if f"{relative_data_path}/results/{output_name}" in dataset_jsons:
                    print(f"Skipping {name} because data already exists")
                    continue
                #if len(fs.glob(path=f"{relative_data_path}/results/{output_name}")) > 0:
                #    print(f"Skipping {name} because data already exists")
                #    continue

                # Input all the csv files
                in_csvs = []
                for trial in range(n_trials):
                    in_csvs.append("--in_csvs")
                    in_csvs.append(f"{dataset[0]}_{seed + trial}.csv")

                # Create the job
                input_folder = f"azureml://datastores/workspaceblobstore/paths/{relative_data_path}/data/{dataset[0]}/"
                out_json_file = f"azureml://datastores/workspaceblobstore/paths/{relative_data_path}/results/{output_name}"

                job = command(
                    inputs=dict(
                        name = name,
                        featurizer_class_name = queue_item["featurizer_class_name"],
                        featurizer_args = str(queue_item["featurizer_args"]),
                        learner_class_name = queue_item["learner_class_name"],
                        learner_args = str(queue_item["learner_args"]),
                        sample_rate = queue_item["sample_rate"],
                        extra_logging = str(queue_item["extra_logging"]),
                        seed = seed + trial,
                        in_folder = Input(
                            type = AssetTypes.URI_FOLDER,
                            path = input_folder,
                            mode = InputOutputModes.RO_MOUNT,
                        )
                    ),
                    outputs = dict(
                        out_json_file = Output(
                            type = AssetTypes.URI_FILE, 
                            path = out_json_file, 
                            mode = InputOutputModes.RW_MOUNT,
                        )
                    ),
                    compute = compute_target,
                    environment = f"{job_env.name}:{job_env.version}",
                    code = "./aml/src/",
                    command = "python aml_run_trial.py " + " ".join(in_csvs) + 
                        " --in_folder ${{inputs.in_folder}} --out_json_file ${{outputs.out_json_file}}" +
                        " --name ${{inputs.name}} --featurizer_class_name ${{inputs.featurizer_class_name}} " +
                        " --featurizer_args \"${{inputs.featurizer_args}}\" --learner_class_name ${{inputs.learner_class_name}} " +
                        " --learner_args \"${{inputs.learner_args}}\" --sample_rate ${{inputs.sample_rate}} " +
                        " --extra_logging \"${{inputs.extra_logging}}\" --seed ${{inputs.seed}} ",
                    experiment_name = "sparse-featurizer-fitting",
                    display_name = name,
                )
                jobs.append(ml_client.jobs.create_or_update(job))


        if len(jobs) > 0:
            # Wait for completion
            print("Waiting for fitting jobs to complete...")
            
            while True:
                status = {}
                for job in jobs:
                    job_status = ml_client.jobs.get(name=job.name).status
                    if job_status not in status:
                        status[job_status] = 0
                    status[job_status] += 1
                
                print(status)

                if all([s in ["Completed", "Failed"] for s in status.keys()]):
                    break
                
                time.sleep(5)
            
            # Check for failures
            if any([s == "Failed" for s in status.keys()]):
                print("WARNING: One or more fitting jobs failed, continuing anyway")
            
            print("Data fitting jobs completed successfully")
        

        # Now merge the results
        print("Merging results...")
        

        # Check if the file already exists in the datastore already
        dataset_jsons = fs.glob(path=f"{relative_data_path}/results/*.json")
        print(f"Found {len(dataset_csvs)} json file results")
        results = []
        for queue_item in self.queue:
            m = hashlib.sha256()
            m.update(json.dumps(queue_item).encode("utf-8"))
            m.update(str(n_trials).encode("utf-8"))
            m.update(str(seed).encode("utf-8"))
            id = m.hexdigest()[:8]
            
            for dataset in self.datasets:
                name = f"{queue_item['name']}_{dataset[0]}_{id}"
                output_name = f"{name}.json"

                #if len(fs.glob(f"{relative_data_path}/results/{output_name}")) == 0:
                #    print(f"WARNING: Results file {output_name} not found in datastore")
                #    continue
                if f"{relative_data_path}/results/{output_name}" not in dataset_jsons:
                    print(f"WARNING: Results file {output_name} not found in datastore")
                    continue
                print(f"Loading results found for {name}...")

                with fs.open(f"{relative_data_path}/results/{output_name}", "r") as f:
                    # Print the file contents
                    data = {"dataset": dataset[0]}
                    for result in json.load(f):
                        entry = copy.deepcopy(data)
                        entry.update(result)
                        results.append(entry)
        
        # Aggregate the results
        group_by = ["dataset"]
        if len(self.queue) > 1:
            group_by.extend( self.queue[0]["extra_logging"].keys() )
        
        results_aggregated, results_merged = self._aggregate_results(results, group_by=group_by)

        return results, results_aggregated, results_merged

        



        
                





            


            
        