import os
import random
import time
from sklearn.linear_model import LogisticRegression
from sparse_benchmark.transforms import BenchmarkStratifiedBagVectorizer, BenchmarkCountVectorizer, BenchmarkTfidfVectorizer, BenchmarkHashingVectorizer, BenchmarkMultiVectorizer
from sparse_benchmark.benchmark import BinaryClassificationBenchmark
from functools import partial
import pandas as pd
from azureml.core import Workspace, Dataset

def run_feature_size_benchmark():
    """Runs the benchmark for the feature size of the BloomVectorizer."""

    # Define the benchmark vectorizers to test
    learner_name = "LogisticRegression"
    learner_args = {
        "max_iter": 3000,
        "penalty": None
    }

    featurizers = {
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
        "StratTfIdf_bag1000_e0.01" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 1000,
                "error_rate": 0.01,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag100_e0.01" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 100,
                "error_rate": 0.01,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
                "ranking_learner_args": learner_args,
            }),
        "StratTfIdf_bag10_e0.01" : 
            ("BenchmarkStratifiedBagVectorizer", {
                "n_bags": 10,
                "error_rate": 0.01,
                "delimiter": '|',
                "task": 'classification',
                "ranking_method": "tfidf-learner",
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
    #n_features = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    n_features = [64, 128]
    random.shuffle(n_features)
    benchmark = BinaryClassificationBenchmark()

    for n_features in n_features:
        print(f"\n--- Adding benchmark for n_features={n_features} ---")
        for name, featurizer_base in featurizers.items():
            print("Adding benchmark for %s..." % name)
            benchmark.add_queue(
                name = name,
                featurizer_class_name = featurizer_base[0],
                featurizer_args = featurizer_base[1],
                learner_class_name = learner_name,
                learner_args = learner_args,
                sample_rate = 0.05,
                extra_logging = {
                    "featurizer" : name,
                    "n_features" : n_features,
                },
            )
    
    # NOTE: Replace these with your own AML subscription details
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")
    compute_target = os.environ.get("COMPUTE_TARGET")
    
    ws = Workspace(subscription_id, resource_group, workspace_name)
    ds = ws.get_default_datastore()

    # Run the benchmark
    print("\n\nStarting benchmark on Azure ML...")
    results_raw, results_aggregated, results_merged = benchmark.run_queue_azure_aml(
        workspace=ws,
        datastore=ds,
        experiment_name="sparse_benchmark",
        compute_target=compute_target,
        data_path="sparse_benchmark\\data",
        output_path="sparse_benchmark\\results",
        n_trials = 5,
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
