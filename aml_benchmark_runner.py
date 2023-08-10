import copy
import os
import random
import time
from sklearn.linear_model import LogisticRegression
from sparse_benchmark.transforms import BenchmarkStratifiedBagVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer, BenchmarkMultiVectorizer
from sparse_benchmark.benchmark import BinaryClassificationBenchmark
from sparse_benchmark.benchmark import standard_datasets
from functools import partial
import pandas as pd
from azureml.core import Workspace, Dataset
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
import numpy as np
import json

def run_benchmark(
        experiment_name,
        n_bags = [1000],
        n_features = [100000],
        error_rates = [0],
        sample_rates = [1.0],
        stratify_learners = [("tfidf-learner", {"max_iter": 3000, "penalty": None})],
        final_learners = [("LogisticRegression", {"max_iter": 3000, "penalty": None})],
    ):
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    featurizers = []
    for n_feature in n_features:
        # StratTfIdf learner
        for stratify_learner_name, stratify_learner_args in stratify_learners:
            for n_bag in n_bags:
                for error_rate in error_rates:
                    featurizers.append(
                        ("StratTfIdf", {
                            "featurizer_class_name": "BenchmarkStratifiedBagVectorizer",
                            "featurizer_args": {
                                "n_bags": n_bag,
                                "error_rate": 0,
                                "delimiter": '|',
                                "task": 'classification',
                                "ranking_method": stratify_learner_name,
                                "ranking_learner_args": stratify_learner_args,
                                "n_features": n_feature,
                            }
                        })
                    )
        
        # Other learners
        featurizers.append(
            ("TfidfVectorizer", {
                "featurizer_class_name": "BenchmarkTfidfVectorizer",
                "featurizer_args": {
                    "delimiter": '|',
                    "n_features": n_feature,
                }
            })
        )
        featurizers.append(
            ("CountVectorizer", {
                "featurizer_class_name": "BenchmarkCountVectorizer",
                "featurizer_args": {
                    "delimiter": '|',
                    "n_features": n_feature,
                }
            })
        )
        featurizers.append(
            ("HashingVectorizer", {
                "featurizer_class_name": "BenchmarkHashingVectorizer",
                "featurizer_args": {
                    "delimiter": '|',
                    "n_features": n_feature,
                }
            })
        )

    # Define the number of features to test
    benchmark = BinaryClassificationBenchmark()

    # Sweep the percent of data to use
    count = 0
    for sample_rate in sample_rates:
        for final_learner_name, final_learner_args in final_learners:
            for name, featurizer_base in featurizers:
                print("Adding benchmark for %s..." % name)
                featurizer_args = copy.deepcopy(featurizer_base["featurizer_args"])
                
                benchmark.add_queue(
                    name = name,
                    featurizer_class_name = featurizer_base["featurizer_class_name"],
                    featurizer_args = featurizer_args,
                    learner_class_name = final_learner_name,
                    learner_args = final_learner_args,
                    sample_rate = sample_rate,
                    extra_logging = {
                        "learner_class_name" : final_learner_name,
                        "featurizer" : name,
                        "n_features" : n_feature,
                        "sample_rate" : sample_rate,
                        "n_bags" : featurizer_args.get("n_bags", None),
                        "error_rate" : featurizer_args.get("error_rate", None),
                    },
                )
                count += 1
    print(f"Added {count * len(standard_datasets)} benchmark runs.")
    input("Press Enter to start...")
    
    # NOTE: Replace these with your own AML subscription details
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")
    compute_target = os.environ.get("COMPUTE_TARGET")

    # Authentication package
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    cpu_cluster = ml_client.compute.get(compute_target)

    root_storage_path = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/workspaceblobstore/"
    relative_data_path = "sparse_benchmark"
    
    # Run the benchmark
    print("\n\nStarting benchmark on Azure ML...")
    results_raw, results_aggregated, results_merged = benchmark.run_queue_azure_aml(
        ml_client=ml_client,
        compute_target=compute_target,
        root_storage_path=root_storage_path,
        relative_data_path=relative_data_path,
        experiment_name=experiment_name,
        n_trials = 10,
        seed = 42,
    )

    # Convert array of dict to df
    with open(f"{experiment_name}_raw.json", "w") as f:
        json.dump(results_raw, f)
    df = pd.DataFrame(results_aggregated)
    df.to_csv(f"{experiment_name}_aggr.csv")

    with open(f"{experiment_name}_merged.json", "w") as f:
        json.dump(results_merged, f)
    df = pd.DataFrame(results_merged)
    df.to_csv(f"{experiment_name}_merged.csv")

    with open(f"{experiment_name}_raw.json", "w") as f:
        json.dump(results_raw, f)
    df.to_csv(f"{experiment_name}_raw.csv")