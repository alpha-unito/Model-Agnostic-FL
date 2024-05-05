from pycox.models import CoxPH
import numpy as np

class CoxPH(CoxPH):
    def __init__(self, net, loss=None, optimizer=None, device=None):
        super().__init__(net, loss, optimizer, device)
    
    def predict_survival_function_(self , X , y):
        event_times_all = np.unique(y['time'])
        follow_up_time = np.max(y['time'])
        event_times = event_times_all[event_times_all < follow_up_time]
        pred = self.predict_surv(X)
        surv = self.predict_surv_df(X)
        Survivals = surv.index.values
        # Interpolate predictions at event times
        pred_interp = np.apply_along_axis(lambda x: np.interp(event_times, Survivals, x), axis=1, arr=pred)
        return pred_interp
    