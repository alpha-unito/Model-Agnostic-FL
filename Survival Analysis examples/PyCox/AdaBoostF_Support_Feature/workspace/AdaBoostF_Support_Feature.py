import argparse
import sys
import numpy as np
import pandas as pd
import wandb
from sksurv.metrics import integrated_brier_score
from support_feature_Dataset import support_feature_Dataset
from sksurv.metrics import concordance_index_censored
import torch
import torchtuples as tt
from adaboost import AdaBoostF
from coxph import CoxPH
from pycox.evaluation import EvalSurv
import random
from sklearn.model_selection import train_test_split

from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True
random.seed(42)

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=10, type=int)
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
    weak_learner = model.get(0)
    durations = y.iloc[:,0].values
    event =  y.iloc[:,1].values
    adaboost_coeff = np.array(adaboost_coeff)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    batch_size = 256
    x_train , x_val , y_train , y_val = train_test_split(X[ids],y.iloc[ids,:],train_size = 0.8 ,random_state = 42 )
    y_train_durations = y_train.iloc[:,0].values
    y_train_event =  y_train.iloc[:,1].values
    y_train = (y_train_durations , y_train_event)
    #Getting the best learning rate lr
    lrfind = weak_learner.lr_finder(x_train, y_train, batch_size, tolerance=50)
    weak_learner.optimizer.set_lr(lrfind.get_best_lr())
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    y_val_durations = y_val.iloc[:,0].values
    y_val_event =  y_val.iloc[:,1].values
    y_val = (y_val_durations,y_val_event)
    val = x_val , y_val
    

    weak_learner.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                     val_data=val, val_batch_size=batch_size)
    _ = weak_learner.compute_baseline_hazards()
    surv = weak_learner.predict_surv_df(X)
    ev = EvalSurv(surv,durations,event, censor_surv='km')
    c_index_value = ev.concordance_td()
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    integrated_brier_score_value = ev.integrated_brier_score(time_grid)
    if LOG_WANDB:
        wandb.log({"weak_train_Corcondance-index": c_index_value,
                 "weak_train_Integrated_Brier_score": integrated_brier_score_value},
                  commit=False)
    
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}

@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    event_indicator = y['event']
    time = y['time']
    adaboost_coeff = np.array(adaboost_coeff)

    rank = int(str(name).split('_')[1]) - 1
    error = []
    c_index = []
    
    durations = y.iloc[:,0].values
    event =  y.iloc[:,1].values
    for idx, weak_learner in enumerate(model.get_estimators()):
        _ = weak_learner.compute_baseline_hazards()
        survs = weak_learner.predict_surv_df(X)
        ev = EvalSurv(survs, durations,event , censor_surv='km')
        c_index_value = ev.concordance_td()
        duration_values = survs.iloc[:,0].index.values 
        survival_duration = np.array([duration_values[np.argmax(survs.iloc[:,i].values <= 0.5)] for i in survs])
        err = np.where(np.logical_or(event_indicator == 1, np.logical_and(event_indicator == 0, survival_duration < time)),
               np.abs(survival_duration - time),
               0)
        
        err = (err-np.min(err))/(np.max(err)-np.min(err))

        error.append(np.dot(adaboost_coeff , err))        #Calculate the weighted error over all the samples for the current weak learner                         
        c_index.append(c_index_value)
        if idx == rank:
            if LOG_WANDB: 
                wandb.log({"weak_validate_Corcondance-index": c_index_value },
                  commit=False)
           
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(np.sum(error))# ou .mean(error) 
    
    return {'errors': error }, {'Concordance-index': c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', adaboost_coeff = 'adaboost_coeff', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    
    X, y = val_loader

    event_indicator = y['event'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    y_pred = model.predict(X)
    c_index_value = concordance_index_censored(event_indicator, time , y_pred)
    c_index_value = c_index_value[0]
    weighted_survivals = model.predict_surv_function(X,y)
    event_times_all = np.unique(y['time'])
    follow_up_time = np.max(y['time'])
    event_times = event_times_all[event_times_all < follow_up_time]
    integrated_brier_score_value = integrated_brier_score(y_structured, y_structured, weighted_survivals, event_times)
    if LOG_WANDB:
         wandb.log({"Model_Corcondance-index": c_index_value , 
                    "Model_Integrated Brier Score": integrated_brier_score_value},
                  commit=False)
  
    return {'Corcondance-index': c_index_value , 'Integrated-Brier-Score' : integrated_brier_score_value}

wandb.init(project="Federated_Learning_support_feature",settings=wandb.Settings(_service_wait=4000))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_support_feature1",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)

in_features = 55
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=CoxPH(net, tt.optim.Adam)),#CoxPHSurvivalAnalysis()),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = support_feature_Dataset()


fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
