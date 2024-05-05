import numpy as np

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
            pred = self.estimator_weights_[i] * pred
            pred = pred.flatten()
            weighted_preds[i,:] = pred
        y_pred = np.sum(weighted_preds ,axis = 0 ) #Calculate the weighted sum of the predictions 
        return y_pred  
    
    def predict_surv_function(self, X: np.ndarray , y: np.ndarray) -> np.ndarray:
        event_times_all = np.unique(y['duration'])
        follow_up_time = np.max(y['duration'])
        event_times = event_times_all[event_times_all < follow_up_time]
        n_times = len(event_times)
        preds = np.empty((self.n_estimators_ , X.shape[0] , n_times))
        for i, clf in enumerate(self.estimators_):
            _ = clf.compute_baseline_hazards()
            preds_interp  =  clf.predict_survival_function_(X , y)
            # Interpolate predictions at event times
            preds[i,:,:] = preds_interp
        weighted_survivals = np.average(preds, axis=0, weights=self.estimator_weights_)
        return weighted_survivals
    