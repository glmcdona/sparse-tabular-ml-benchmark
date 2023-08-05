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
        "StratTfIdf_bag10000" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 10000,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag5000" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 5000,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag1000" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 1000,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag100" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 100,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag10" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 10,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratChi_bag1000" :
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 1000,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "chi",
                "ranking_learner_args": learner_args,
            }),
        "StratChi_bag100" :
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 100,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "chi",
                "ranking_learner_args": learner_args,
            }),
        "StratChi_bag10" :
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 10,
                "error_rate": 0,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "chi",
                "ranking_learner_args": learner_args,
            }),
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
    
    all_results_raw = []
    all_results_aggr = []

    # Define the number of features to test
    benchmark = BinaryClassificationBenchmark()

    # Sweep the percent of data to use
    #for sample_rate in [0.1, 0.25, 0.5, 1.0]:
    for sample_rate in [0.1, 0.10]:
        n_features = [2000, 10000, 50000, 500000]

        for n_features in n_features:
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
                    },
                )
    
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
    df.to_csv("aml_results_aggr.csv")
    df = pd.DataFrame(results_raw)
    df.to_csv("aml_results_raw.csv")
    df = pd.DataFrame(results_merged)
    df.to_csv("aml_results_merged.csv")



if __name__ == "__main__":
    # Run the benchmark
    run_feature_size_benchmark()
