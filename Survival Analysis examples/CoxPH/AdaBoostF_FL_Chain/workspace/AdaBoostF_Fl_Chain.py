import argparse
import sys
import numpy as np
import wandb
from sksurv.metrics import integrated_brier_score
from coxph import CoxPH
from Fl_Chain_Dataset import Fl_Chain_Dataset
from sksurv.metrics import concordance_index_censored

from adaboost import AdaBoostF
import random

from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True
random.seed(42)

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=50, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--server", default='localhost', type=str, help="server address")
args = parser.parse_args()

random_state = np.random.RandomState(args.seed)

client_id = 'api'
cert_dir = '../cert'

task_interface = TaskInterface()
 

@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def train_adaboost(model, train_loader, device, optimizer, adaboost_coeff, name):
    X, y = train_loader
    event_indicator = y['death'].astype(bool)
    futime = y['futime']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, futime)], dtype=[('death', bool), ('futime', float)])
    adaboost_coeff = np.array(adaboost_coeff)
    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X.iloc[ids], y_structured[ids])
    preds = weak_learner.predict_survival_function_(X , y)
    event_times = np.unique(y['futime'])
    follow_up_time = np.max(y['futime'])
    event_times = event_times[event_times < follow_up_time]
    y_pred = weak_learner.predict(X)
    c_index_value = concordance_index_censored(y['death'], y['futime'], y_pred)
    c_index_value = c_index_value[0]
    integrated_brier_score_value = integrated_brier_score(y_structured, y_structured, preds, event_times)
    if LOG_WANDB:
        wandb.log({"Train_Corcondance-index": c_index_value,
                 "Train_Integrated_Brier_score": integrated_brier_score_value},
                  commit=False)
  
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}

@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    adaboost_coeff = np.array(adaboost_coeff)
    rank = int(str(name).split('_')[1]) - 1
    error = []
    c_index = []
    for idx, weak_learner in enumerate(model.get_estimators()):
        pred = weak_learner.predict(X)
        c_index_value = concordance_index_censored(y['death'] , y['futime'] ,pred )
        c_index_value = c_index_value[0]
        survs = weak_learner.predict_survival_function(X)
        Survivals = survs[0].x
        survival_times = np.array([Survivals[np.argmax(surv.y <= 0.5)] for surv in survs])
        err = np.where(np.logical_or(y['death'] == 1, np.logical_and(y['death'] == 0, survival_times < y['futime'])),
               np.abs(survival_times - y['futime']),
               0)
        err = (err-np.min(err))/(np.max(err)-np.min(err))
        error.append(np.dot(adaboost_coeff , err))        #Calculate the weighted error over all the samples for the current client                          
        c_index.append(c_index_value)
       
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(np.sum(error))# ou .mean(error) 
    
    return {'errors': error }, {'Concordance-index': c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', adaboost_coeff = 'adaboost_coeff', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    
    X, y = val_loader
    event_indicator = y['death'].astype(bool)
    futime = y['futime']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, futime)], dtype=[('death', bool), ('futime', float)])
    y_pred = model.predict(X)
    c_index_value = concordance_index_censored(y['death'], y['futime'], y_pred)
    c_index_value = c_index_value[0]
    weighted_survivals = model.predict_surv_function(X,y)
    event_times_all = np.unique(y['futime'])
    follow_up_time = np.max(y['futime'])
    event_times = event_times_all[event_times_all < follow_up_time]
    integrated_brier_score_value = integrated_brier_score(y_structured, y_structured, weighted_survivals, event_times)
    if LOG_WANDB:
         wandb.log({"Model Validation Corcondance index": c_index_value,
                    "Model Validation Integrated Brier_score": integrated_brier_score_value},
                  commit=False)
  
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}

wandb.init(project="Federated_Learning_Fl_Chain",settings=wandb.Settings(_service_wait=600))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_Fl_Chain",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=CoxPH()),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = Fl_Chain_Dataset()


fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
