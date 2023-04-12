# OpenFL-x - OpenFederatedLearning-extended

OpenFederatedLearning-extended (OpenFL-x) is an **open-source extension** of [Intel® OpenFL](https://github.com/securefederatedai/openfl) supporting *federated bagging and boosting of any ML model*. The software is entirely Python-based and comes with an extensive set of examples, as described belolw, exploiting [SciKir-Learn](https://scikit-learn.org/stable/) models. It has been successfully tested on x86_64, ARM and RISC-V platforms.



## Installation

It is highly recommended to create a **virtual environment** prior to install OpenFL-x (with both [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/)) to avoid compatibility issues with existing Python software already installed on the system.
Furthermore, since OpenFL-x is and extended version of OpenFL, it integrates all the features of the base framework plus the federated bagging and boosting functionalities: it is then suggested to **not install both softwares in the same virtual environment**, since this can lead to issues in the working of both softwares.

To install OpenFL-x simply clone this repository and run `pip install`:
```
git clone https://github.com/alpha-unito/OpenFL-extended.git
cd OpenFL-extended
pip install .
```
If this procedure completes successfully, you now have access to all the base feature of OpenFL plus the distributed bagging and boosting functinoalities. Enjoy!



## Getting Started

The quickest way to test OpenFL-x is to execute the examples available in the `boosting-examples` folder. There are many of them available to be run out-of-the-box, each emploing a different dataset and a different number of participant in the federation:

| Name                     | Algorithm          | Dataset      | # Train samples | # Test samples | # labels | # Features | # Envoy | ML model                 |
| :----------------------- | :----------------- | :----------- | --------------: | -------------: | -------: | ---------: | ------: | :----------------------- |
| `AdaBoostF_Adult`        | Federated boosting | Adult        | 30,132          | 15,060         | 2        | 14         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_Iris`         | Federated boosting | Iris         | 35              | 15             | 3        | 4          | 2       | `DecisionTreeClassifier` |
| `AdaBoostF_forestcover`  | Federated boosting | ForestCover  | 250,000         | 245,141        | 2        | 54         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_krvskp`       | Federated boosting | KrvsKp       | 2,557           | 639            | 2        | 36         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_letter`       | Federated boosting | Letter       | 16,000          | 4,000          | 26       | 16         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_pendigits`    | Federated boosting | Pendigits    | 7,494           | 3,498          | 10       | 16         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_sat`          | Federated boosting | Sat          | 4,435           | 2,000          | 8        | 36         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_segmentation` | Federated boosting | Segmentation | 209             | 2,099          | 7        | 19         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_splice`       | Federated boosting | Splice       | 2,552           | 638            | 3        | 61         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_vowel`        | Federated boosting | Vowel        | 792             | 198            | 11       | 27         | 10      | `DecisionTreeClassifier` |
| `AdaBoostF_vehicle`      | Federated boosting | Vehicle      | 677             | 169            | 4        | 18         | 10      | `DecisionTreeClassifier` |
| `RandomForest_Iris`      | Federated bagging 	| Iris         | 35              | 15             | 3        | 4          | 2       | `DecisionTreeClassifier` |

Obviously, each example can be customised by the user by changing the data distribution across the envoys, the number of envoys itself, the ML model used as weak learner, and clearly also the aggregation algorithm used by the federation. More information on how to run and personalise the examples are available in the `boosting-examples` folder.

Example results of the execution of these test and their variations are freely available on [WandB](https://wandb.ai/gmittone/AdaBoost.F?workspace=user-gmittone). During the experimentation many SciKit-Learn classifiers have been used as weak learners, such as `ExtremelyRandomizedTree`, `RidgeLinearRegression`, `MultiLayerPerceptron`, `KNearestNeighbors`, `GaussianNaiveBayes`, and `DecisionTree`.



## Performance optimisations



## Aggregation algorithm




## Publication

This work is currently under review at the [29th International European Conference on Parallel and Distributed Computing](https://2023.euro-par.org).
The paper's citation and link will be provided as soon as they became available.

A pre-print version of the paper is available on [arXiv](https://arxiv.org/abs/2303.04906).


## Contacts

This software is developed and maintained by [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) (gianluca.mittone@unito.it).

This software offers the possibility to experiment with basic Distributed Bagging and the Distributed Boosting method called AdaBoost.F developed by Polato et al.

[Polato, Mirko, Roberto Esposito, and Marco Aldinucci. "Boosting the federation: Cross-silo federated learning without gradient descent." 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022.]


