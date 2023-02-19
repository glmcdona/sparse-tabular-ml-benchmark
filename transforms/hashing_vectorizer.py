from .base import BaseBenchmarkPipeline
from sklearn.feature_extraction.text import HashingVectorizer

class BenchmarkHashingVectorizer(BaseBenchmarkPipeline):
    def __init__(self, delimiter='|', task='classification', n_features=100000):
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
        super(BenchmarkHashingVectorizer, self).__init__(delimiter, task)
        self.featurizer = None
        self.n_features = n_features
    

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
        featurizer = HashingVectorizer(
            tokenizer = lambda x: x.split(self.delimiter),
            n_features = self.n_features,
            token_pattern = None,
        )
        self.featurizer = featurizer.fit(X, y)

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
        if self.featurizer is None:
            raise ValueError('Pipeline is not fitted yet.')
        return self.featurizer.transform(X)
        
    
    def get_size_in_bytes(self):
        """Returns the full size of the featurizer in bytes.

        Returns:
            int: Size of the featurizer steps in bytes.
        """
        if self.featurizer is None:
            raise ValueError('Pipeline is not fitted yet.')
        
        # Hashing vectorizer has no size needed to be stored
        # within the transform.
        return 0
    

    def get_num_features(self):
        """Returns the number of features used by the featurizer.

        Returns:
            int: Number of features in the featurizer.
        """
        if self.featurizer is None:
            raise ValueError('Pipeline is not fitted yet.')
        return 0