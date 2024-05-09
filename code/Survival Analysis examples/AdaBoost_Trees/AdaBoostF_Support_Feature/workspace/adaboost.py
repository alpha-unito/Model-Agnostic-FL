import numpy as np
from sksurv.functions import StepFunction

class AdaBoostF:
    def __init__(self, base_estimator):
        self.estimators_ = [base_estimator]
        self.n_estimators_ = 1
        self.estimator_weights_ = [1]

    def get_estimators(self):
        return self.estimators_

    def add(self, weak_learner, coeff):
        self.estimators_.append(weak_learner)
        self.estimator_weights_ = np.append(self.estimator_weights_, coeff)
        self.n_estimators_ += 1
    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        weighted_preds = np.zeros((self.n_estimators_ ,  np.shape(X)[0])) #Store the weighted predictions for each estimator
        for i, clf in enumerate(self.estimators_):
            pred = clf.predict(X)
            weighted_preds[i,:] = self.estimator_weights_[i] * pred
        y_pred = np.sum(weighted_preds ,axis = 0 ) #Calculate the weighted sum of the predictions 
        return y_pred  
    
    def predict_surv_function(self, X: np.ndarray , y: np.ndarray) -> np.ndarray:
        event_times_all = np.unique(y['time'])
        follow_up_time = np.max(y['time'])
        event_times = event_times_all[event_times_all < follow_up_time]
        n_times = len(event_times)
        preds = np.empty((len(self.estimators_), X.shape[0], n_times))

        # Compute predictions for each estimator
        for i, clf in enumerate(self.estimators_):
            survs = clf.predict_survival_function(X , return_array = True)
            Survivals = clf.unique_times_
            # Interpolate predictions at event times
            pred_interp = np.apply_along_axis(lambda x: np.interp(event_times, Survivals, x), axis=1, arr=survs)
            preds[i,:,:] = pred_interp

        # Compute the weighted average of predicted survival functions
        weighted_survivals = np.average(preds, axis=0, weights=self.estimator_weights_)
        return weighted_survivals
    