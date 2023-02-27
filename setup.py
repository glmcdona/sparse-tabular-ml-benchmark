from setuptools import setup, find_packages

setup(
    name='sparse_benchmark',
    version='0.1',
    description='Sparse benchmark dataset',
    author='Geoff McDonald',

    packages=find_packages(include=["sparse_benchmark"]),

    install_requires=[
        "pandas",
        "azureml-core",
        "azureml-pipeline-core",
        "azureml-pipeline",
        "sklearn",
        "stratified_vectorizer",
        "fsspec",
    ]
)
