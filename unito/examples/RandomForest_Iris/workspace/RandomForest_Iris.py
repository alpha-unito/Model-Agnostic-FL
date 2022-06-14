import numpy as np
from openfl.interface.interactive_api.experiment import TaskInterface, ModelInterface
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

from IrisDataset import IrisDataset
from unito.examples.RandomForest_Iris.director.random_forest import MyRandomForestClassifier
from unito.openfl_ext.experiment import FLExperiment
from unito.openfl_ext.federation import Federation

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = '../cert'
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
    try:
        if model is not None:
            X, y = val_loader
            pred = model.predict(X)
            metric = accuracy_score(pred, y, normalize=True)
        else:
            print("Model not found")
            metric = 0
    except NotFittedError:
        print("Model is not yet fit")
        metric = 0

    return {'accuracy': metric}


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50052', tls=False)
# @TODO: the serializer plugin should be read from the plan
fl_experiment = FLExperiment(federation=federation, experiment_name="RandomForest_Iris",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer')
model_interface = ModelInterface(model=MyRandomForestClassifier(), optimizer=None,
                                 framework_plugin='unito.openfl_ext.generic_adapter.GenericAdapter')
federated_dataset = IrisDataset(random_state=random_state)

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=3,  # @TODO: It only make sense to make one round though
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
    default=False
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
