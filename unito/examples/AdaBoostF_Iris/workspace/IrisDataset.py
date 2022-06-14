from openfl.interface.interactive_api.experiment import DataInterface
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisDataset(DataInterface):
    def __init__(self, random_state=42, **kwargs):
        super().__init__(**kwargs)

        iris = load_iris()
        X, y = iris.data, iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=random_state,
                                                                                shuffle=True)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return self.X_train, self.y_train

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return self.X_test, self.y_test

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.X_train)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.X_test)
