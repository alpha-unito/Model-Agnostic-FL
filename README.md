# OpenFL-x - OpenFederatedLearning-extended

OpenFederatedLearning-extended (OpenFL-x) is an **open-source extension** of [Intel® OpenFL](https://github.com/securefederatedai/openfl) supporting *federated bagging and boosting of any ML model*. The software is entirely Python-based and comes with an extensive set of examples. It has been successfully tested on x86_64, ARM and RISC-V platforms.



## Installation

It is highly recommended to create a virtual environment prior to install OpenFL-x (with both [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/)) to avoid compatibility issues with existing Python software already installed on the system.
Furthermore, since OpenFL-x is and extended version of OpenFL, it integrates all the features of the base framework plus the federated bagging and boosting functionalities: it is then suggested to not install both softwares in the same virtual environment, since this can lead to issues in the working of both softwares.

To install OpenFL-x simply clone this repository and run `pip install`:
```
git clone https://github.com/alpha-unito/OpenFL-extended.git
cd OpenFL-extended
pip install .
```
If this procedure completes successfully, you now have access to all the base feature of OpenFL plus the distributed bagging and boosting functinoalities. Enjoy!



## Getting Started

The quickest way to test OpenFL-x is to execute the examples available in the `boosting-examples` folder. There are many of them available to be run out-of-the-box, each emploing a different dataset and a different number of participant in the federation.



## Publication

This work is currently under review at the [29th International European Conference on Parallel and Distributed Computing](https://2023.euro-par.org).

A pre-print version of the paper is available on [arXiv](https://arxiv.org/abs/2303.04906).

The paper's citation and link will be provided as soon as they became available.


## Contacts

This software is developed and maintained by [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) (gianluca.mittone@unito.it).

This software offers the possibility to experiment with basic Distributed Bagging and the Distributed Boosting method called AdaBoost.F developed by Polato et al.

[Polato, Mirko, Roberto Esposito, and Marco Aldinucci. "Boosting the federation: Cross-silo federated learning without gradient descent." 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022.]


