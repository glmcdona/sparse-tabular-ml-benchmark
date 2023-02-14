from base import BaseBenchmarkPipeline
from bloombag import BloomVectorizer

class BenchmarkBloomVectorizer(BaseBenchmarkPipeline):
    def __init__(self, n_bags=5, error_rate=0.01):
        pass

    def get_features(self, X_train, y_train, n_features, learner, delimiter, task):
        """Fits the pipeline to the training data.

        Args:
            X_train (array): Array of strings. This will be the training data
                used to fit the pipeline. Each value is a string of the form
                "feature1,feature2,...,featureN" delimited by the specified
                delimiter.
            y_train (array): An array of 1 and 0 labels for 'classification'
                task, or an array of floats for 'regression' task.
            learner (_type_): A Scikit-learn estimator that is going to be used
                to fit the pipeline. This can be optionally used.
            delimiter (str): A string that specifies the delimiter used in
            task (str): A string that specifies the task. This can be either
                'classification' or 'regression'.

        Returns:
            numpy.ndarray: An array of features that will be passed to the
                learner.
        """
        featurizer = BloomVectorizer(
            tokenizer = lambda x: x.split(delimiter),
            n_features = n_features,
            n_bags = self.n_bags,
            error_rate = self.error_rate,
            token_pattern = None,
        )
