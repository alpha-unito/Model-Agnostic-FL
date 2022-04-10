import numpy as np
from openfl.interface.interactive_api.experiment import TaskInterface, ModelInterface
from sklearn.metrics import accuracy_score

from IrisDataset import IrisDataset
from MyModel import MyRandomForestClassifier
from unito.openfl_ext.experiment import FLExperiment
from unito.openfl_ext.federation import Federation

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = 'cert'
director_node_fqdn = 'localhost'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer')
def train(model, train_loader, device, optimizer):
    X, y = train_loader
    model.fit(X, y)
    pred = model.predict(X)
    metric = accuracy_score(pred, y, normalize=True)

    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    X, y = val_loader

    pred = model.predict(X)
    metric = accuracy_score(pred, y, normalize=True)

    return {'accuracy': metric}


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="federated_RF",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer')  # Perché non lo prende dal plan?
model = MyRandomForestClassifier()
model_interface = ModelInterface(model=model, optimizer=None,
                                 framework_plugin='openfl_ext.generic_adapter.GenericAdapter')
federated_dataset = IrisDataset(random_state=random_state)

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=10,
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
    default=False
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
