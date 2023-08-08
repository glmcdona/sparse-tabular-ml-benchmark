import copy
import os
import random
import time
from sklearn.linear_model import LogisticRegression
from sparse_benchmark.transforms import BenchmarkStratifiedBagVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer, BenchmarkMultiVectorizer
from sparse_benchmark.benchmark import BinaryClassificationBenchmark
from functools import partial
import pandas as pd
from azureml.core import Workspace, Dataset
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    learner_name = "LogisticRegression"

    learner_args = {
        "max_iter": 3000,
        "penalty": None,
    }

    featurizers = {
        "TfidfVectorizer" :
            ("BenchmarkTfidfVectorizer", {
                "delimiter": '|',
            }),
        "HashingVectorizer" :
            ("BenchmarkHashingVectorizer", {
                "delimiter": '|',
            }),
        "CountVectorizer" :
            ("BenchmarkCountVectorizer", {
                "delimiter": '|',
            }),
    }

    for n_bags in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000]:
        featurizers[f"StratTfIdf_bag{n_bags}"] = (
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": n_bags,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            })
        )
    
    all_results_raw = []
    all_results_aggr = []

    # Define the number of features to test
    benchmark = BinaryClassificationBenchmark()

    # Sweep the percent of data to use
    count = 0
    for sample_rate in [0.1, 1.0]:
        for n_features in [100000]:
            print(f"\n--- Adding benchmark for n_features={n_features} ---")
            for name, featurizer_base in featurizers.items():
                print("Adding benchmark for %s..." % name)
                featurizer_args = copy.deepcopy(featurizer_base[1])
                featurizer_args["n_features"] = n_features
                
                benchmark.add_queue(
                    name = name,
                    featurizer_class_name = featurizer_base[0],
                    featurizer_args = featurizer_args,
                    learner_class_name = learner_name,
                    learner_args = learner_args,
                    sample_rate = sample_rate,
                    extra_logging = {
                        "featurizer" : name,
                        "n_features" : n_features,
                        "sample_rate" : sample_rate,
                        "n_bags" : featurizer_args.get("n_bags", None),
                        "error_rate" : featurizer_args.get("error_rate", None),
                    },
                )
                count += 1
    print(f"Added {count} benchmarks.")
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
        experiment_name="sparse_benchmark",
        n_trials = 10,
        seed = 42,
    )

    # Convert array of dict to df
    df = pd.DataFrame(results_aggregated)
    df.to_csv("aml_results_nbags_aggr.csv")
    df = pd.DataFrame(results_raw)
    df.to_csv("aml_results_nbags_raw.csv")
    df = pd.DataFrame(results_merged)
    df.to_csv("aml_results_nbags_merged.csv")



if __name__ == "__main__":
    # Run the benchmark
    run_feature_size_benchmark()
