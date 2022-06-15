import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostF(AdaBoostClassifier):
    def __init__(self, base_estimator):
        super().__init__(n_estimators=1, base_estimator=base_estimator)

    def add(self, weak_learner, coeff):
        self.estimators_.append(weak_learner)
        self.estimator_weights_ = np.append(self.estimator_weights_, coeff)

    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self
