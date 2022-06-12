import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class MyRandomForestClassifier(AdaBoostClassifier):
    def __init__(self):
        super(MyRandomForestClassifier, self).__init__(n_estimators=1,
                                                       base_estimator=DecisionTreeClassifier(max_depth=2))

    def add(self, weak_learner, coeff):
        self.estimators_.append(weak_learner)
        self.estimator_weights_ = np.append(self.estimator_weights_, coeff)

    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self
