

def BaseBenchmarkPipeline():
    def __init__(self, delimiter, task):
        """Initializes the pipeline.

        Args:
            delimiter (str): A string that specifies the delimiter used in
                the input data.
            task (str): A string that specifies the task. This can be either
                'classification' or 'regression'.
        """
        self.delimiter = delimiter
        self.task = task

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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def get_size_in_bytes(self):
        """Returns the full size of the featurizer in bytes.

        Returns:
            int: Size of the featurizer steps in bytes.
        """
        raise NotImplementedError
    
    def get_num_features(self):
        """Returns the number of features used by the featurizer.

        Returns:
            int: Number of features in the featurizer.
        """
        raise NotImplementedError