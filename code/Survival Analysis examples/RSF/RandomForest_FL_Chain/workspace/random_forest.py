from sksurv.ensemble import RandomSurvivalForest
import numpy as np

class MyRandomSurvivalForest(RandomSurvivalForest):
    def __init__(self):
        super().__init__(n_estimators=5)    
    def predict_survival_function_(self , X , y):
        event_times_all = np.unique(y['futime'])
        follow_up_time = np.max(y['futime'])
        event_times = event_times_all[event_times_all < follow_up_time]
        n_times = len(event_times)
        preds = np.empty((len(self.estimators_) , X.shape[0] , n_times))
        for i, clf in enumerate(self.estimators_):
            survs = clf.predict_survival_function(X)
            Survivals = survs[0].x
            pred = clf.predict_survival_function(X ,return_array = True)
            # Interpolate predictions at event times
            pred_interp = np.apply_along_axis(lambda x: np.interp(event_times, Survivals, x), axis=1, arr=pred)
            preds[i,:,:] = pred_interp
        weighted_survivals = np.average(preds, axis=0, weights=np.ones(len(self.estimators_)))
        return weighted_survivals
                 
