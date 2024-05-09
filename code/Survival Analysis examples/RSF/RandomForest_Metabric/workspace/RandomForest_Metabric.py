import argparse
import wandb
import numpy as np
from sklearn.exceptions import NotFittedError
from sksurv.metrics import integrated_brier_score
from metabric_Dataset import metabric_Dataset
from sksurv.metrics import concordance_index_censored
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation
from random_forest import MyRandomSurvivalForest


LOG_WANDB = True
parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=50, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--server", default='localhost', type=str, help="server address")
args = parser.parse_args()

random_state = np.random.RandomState(args.seed)

client_id = 'api'
cert_dir = '../cert'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer' )
def train(model, train_loader, device, optimizer):
    X, y = train_loader
    event_indicator = y['event']
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    model.fit(X, y_structured)
    y_pred = model.predict(X)
    survivals = model.predict_survival_function_(X,y)
    c_index_value = concordance_index_censored(y['event'], y['time'], y_pred)
    c_index_value = c_index_value[0]
    event_times_all = np.unique(y['time'])
    follow_up_time = np.max(y['time'])
    event_times = event_times_all[event_times_all < follow_up_time]
    integrated_brier_score_value = integrated_brier_score(y_structured, y_structured, survivals , event_times)
    if LOG_WANDB:
         wandb.log({"Model Train_Corcondance-index": c_index_value , 
                    "Model Train_Integrated Brier Score": integrated_brier_score_value},
                  commit=False)
  
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    try:
        if model is not None:
            X, y = val_loader
            event_indicator = y['event']
            time = y['time']
            y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
            y_pred = model.predict(X)
            survivals = model.predict_survival_function_(X,y)
            c_index_value = concordance_index_censored(y['event'], y['time'], y_pred)
            c_index_value = c_index_value[0]
            event_times_all = np.unique(y['time'])
            follow_up_time = np.max(y['time'])
            event_times = event_times_all[event_times_all < follow_up_time]
            integrated_brier_score_value = integrated_brier_score(y_structured, y_structured, survivals , event_times)
            if LOG_WANDB:
                wandb.log({"Model Validate_Corcondance-index": c_index_value , 
                   "Model Validate_Integrated Brier Score": integrated_brier_score_value},
                  commit=False)
        else:
            print("Model not found")
            c_index_value = 0
    except NotFittedError:
        print("Model is not yet fit")
        c_index_value = 0
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}


wandb.init(project="Federated_Learning_metabric_2",settings=wandb.Settings(_service_wait=120))

federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="RandomForest_metabric",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=MyRandomSurvivalForest(),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = metabric_Dataset()

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
