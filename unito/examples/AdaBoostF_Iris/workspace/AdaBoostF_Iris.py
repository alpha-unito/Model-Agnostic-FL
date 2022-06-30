import numpy as np
from openfl.interface.interactive_api.experiment import ModelInterface
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from IrisDataset import IrisDataset
from unito.examples.AdaBoostF_Iris.director.adaboost import AdaBoostF
from unito.openfl_ext.experiment import FLExperiment, TaskInterface
from unito.openfl_ext.federation import Federation

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = '../cert'
director_node_fqdn = 'localhost'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff')
def train_adaboost(model, train_loader, device, optimizer, adaboost_coeff):
    weak_learner = model.get(0)

    X, y = train_loader
    X = np.array(X)
    y = np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X[ids], y[ids])

    pred = weak_learner.predict(X)
    metric = accuracy_score(y, pred)

    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff')
def validate_weak_learners(model, val_loader, device, adaboost_coeff):
    X, y = val_loader
    X = np.array(X)
    y = np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)

    error = []
    miss = []
    for weak_learner in model.get_estimators():
        pred = weak_learner.predict(X)
        error.append(sum(adaboost_coeff[y != pred]))
        # error.append(
        #    sum([coeff if x_pred != x_true else 0 for x_pred, x_true, coeff in zip(pred, y, adaboost_coeff)]))
        # miss.append([1 if x_pred != x_true else 0 for x_pred, x_true in zip(pred, y)])
        miss.append(y != pred)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(sum(adaboost_coeff))

    return {'errors': error}, {'misprediction': miss}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    X, y = val_loader
    pred = model.predict(np.array(X))
    f1 = f1_score(y, pred, average="micro")

    return {'F1 score': f1}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaBoostF_Iris",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer')
model_interface = ModelInterface(model=AdaBoostF(base_estimator=DecisionTreeClassifier(), n_classes=3),
                                 optimizer=None,
                                 framework_plugin='unito.openfl_ext.generic_adapter.GenericAdapter')
federated_dataset = IrisDataset(random_state=random_state)

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=300,
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
    default=False
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
