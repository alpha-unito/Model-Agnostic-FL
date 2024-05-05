from sksurv.linear_model import CoxPHSurvivalAnalysis 
import numpy as np

class CoxPH(CoxPHSurvivalAnalysis ):
    def __init__(self):
        super().__init__(alpha = 0.01)
    
    def predict_survival_function_(self , X , y):
        event_times_all = np.unique(y['futime'])
        follow_up_time = np.max(y['futime'])
        event_times = event_times_all[event_times_all < follow_up_time]
        Survivals = self.unique_times_
        pred = self.predict_survival_function(X ,return_array = True)
        # Interpolate predictions at event times
        pred_interp = np.apply_along_axis(lambda x: np.interp(event_times, Survivals, x), axis=1, arr=pred)
        return pred_interp
    