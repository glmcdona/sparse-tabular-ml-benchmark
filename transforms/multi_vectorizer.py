import numpy as np
from .base import BaseBenchmarkPipeline
from bloombag import StratifiedBagVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

class BenchmarkMultiVectorizer(BaseBenchmarkPipeline):
    def __init__(self, vectorizer_classes=[], n_features=100000):
        """Initializes the pipeline.

        Args:
            delimiter (str): A string that specifies the delimiter used in
                the input data.
            task (str): A string that specifies the task. This can be either
                'classification' or 'regression'.
            n_bags (int): Number of bloom filters to use to create count bags.
            error_rate (float): The desired error rate for the bloom filters.
            n_features (int): Number of features to use sorted by occurence
                count in the training data.
        """
        super(BenchmarkMultiVectorizer, self).__init__("|", "classification")

        self.pipeline = []
        for vectorizer_class in vectorizer_classes:
            self.pipeline.append(vectorizer_class(n_features=n_features))
    

    def fit(self, X, y):
        """Fits the pipeline to the training data.

        Args:
            X (array): Array of strings. This will be the training data
                used to fit the pipeline. Each value is a string of the form
                "feature1,feature2,...,featureN" delimited by the specified
                delimiter.
            y (array): An array of 1 and 0 labels for 'classification'
                task, or an array of floats for 'regression' task.
        """
        for vectorizer in self.pipeline:
            vectorizer.fit(X, y)

    def transform(self, X):
        """Transforms the training data into a feature matrix.

        Args:
            x (array): Array of strings. This will be the training data
                used to fit the pipeline. Each value is a string of the form
                "feature1,feature2,...,featureN" delimited by the specified
                delimiter.

        Returns:
            numpy.ndarray: An array of features that will be passed to the
                learner.
        """
        if self.pipeline is None:
            raise ValueError('Pipeline is not fitted yet.')
        
        # Iterate through the steps and transform the data, and
        # union the results together
        return_result = None
        for vectorizer in self.pipeline:
            result = vectorizer.transform(X)
            if type(result) is not np.ndarray:
                result = result.toarray()
            if return_result is None:
                return_result = result
            else:
                return_result = np.concatenate((return_result, result), axis=1)

        return return_result
    
    def get_size_in_bytes(self):
        """Returns the full size of the featurizer in bytes.

        Returns:
            int: Size of the featurizer steps in bytes.
        """
        if self.pipeline is None:
            raise ValueError('Pipeline is not fitted yet.')

        # Iterate through the steps and sum the size of each featurizer
        size = 0
        for vectorizer in self.pipeline:
            size += vectorizer.get_size_in_bytes()
        return size
    

    def get_num_features(self):
        """Returns the number of features used by the featurizer.

        Returns:
            int: Number of features in the featurizer.
        """
        if self.pipeline is None:
            raise ValueError('Pipeline is not fitted yet.')
        
        # Iterate through the steps and sum the number of features
        num_features = 0
        for vectorizer in self.pipeline:
            num_features += vectorizer.get_num_features()
        return num_features
        
