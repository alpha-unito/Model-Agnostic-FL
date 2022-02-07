import openfl_ext.native as fx
from openfl_ext.fl_model import FederatedModel
from openfl.federated.data.federated_data import FederatedDataSet

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from MyModel import MyRandomForestClassifier

random_state = np.random.RandomState(31415)

collaborator_list = [str(i) for i in range(4)]

fx.init(col_names=collaborator_list)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=random_state,
                                                    shuffle=False)

fl_data = FederatedDataSet(X_train, y_train, X_test, y_test, batch_size=100000, num_classes=3)
fl_model = FederatedModel(build_model=MyRandomForestClassifier, data_loader=fl_data)
experiment_collaborators = {col_name: col_model for col_name, col_model
                            in zip(collaborator_list, fl_model.setup(len(collaborator_list)))}
final_fl_model = fx.run_experiment(experiment_collaborators,
                                   override_config={"aggregator.settings.rounds_to_train": 5,
                                                    "aggregator.template": "openfl_ext.aggregator.Aggregator",
                                                    "collaborator.template": "openfl_ext.collaborator.Collaborator",
                                                    "compression_pipeline.template": "openfl_ext.generic_pipeline.GenericPipeline"},
                                   nn=False)
