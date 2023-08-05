from setuptools import setup, find_packages

setup(
    name='sparse_benchmark',
    version='0.1',
    description='Sparse benchmark dataset',
    author='Geoff McDonald',

    packages=find_packages(include=["sparse_benchmark"]),

    install_requires=[
        "pandas",
        "azure-ai-ml",
        "azure-identity",
        "scikit-learn",
        "azureml-fsspec",
        "mltable"
    ],
    extras_require={
        'stratified-vectorizer': [
            'stratified_vectorizer @ git+https://github.com/glmcdona/stratified-vectorizer.git'
        ]
    }
)
