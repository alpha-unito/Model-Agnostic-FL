import numpy as np
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface

from unito.openfl_ext.experiment import FLExperiment
from unito.openfl_ext.federation import Federation

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from MyModel import MyRandomForestClassifier
from sklearn.metrics import accuracy_score

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = 'cert'
director_node_fqdn = 'localhost'

task_interface = TaskInterface()


class IrisDataset(DataInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        iris = load_iris()
        X, y = iris.data, iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, train_size=0.7,
                                                                                random_state=random_state,
                                                                                shuffle=False)

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


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device')
def train(model, train_loader, device):
    X, y = train_loader.get_train_loader()

    model.fit(train_loader, X, y)

    return model


@task_interface.register_fl_task(model='net_model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    X, y = val_loader.get_val_loader()

    model.predict(X)
    metric = accuracy_score(X, y, normalize=True)

    return metric


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50051', tls=False)
framework_adapter = 'openfl_ext.generic_adapter.GenericAdapter'
federated_dataset = IrisDataset()
model_interface = ModelInterface(model=MyRandomForestClassifier, optimizer=None, framework_plugin=framework_adapter)
fl_experiment = FLExperiment(federation=federation, experiment_name="federated_RF",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer') # Perché non lo prende dal plan?

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=5,
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
    default=False
)

fl_experiment.stream_metrics(tensorboard_logs=False)
