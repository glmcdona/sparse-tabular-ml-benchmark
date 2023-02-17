from loaders import loader_newsgroup_binary

def BinaryClassificationDatasets():
    def __init__(self, datasets=None):
        """Initializes the dataset.
        
        Args:
            dataset_names (list): A list of dataset names to use. If None, all
                datasets will be used.
        """
        if datasets is None:
            datasets = {
                "newsgroups": loader_newsgroup_binary
            }
            
        self.datasets = datasets

    def get_next_benchmark(self):
        """Generator returns the next benchmark to run.
        
        Returns:
            tuple: A tuple containing the name of the benchmark, dataframe
                containing "features" and "labels" columns.
        """
        for name, loader in self.datasets.items():
            yield name, loader()