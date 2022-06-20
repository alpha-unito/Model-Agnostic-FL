import numpy as np
from openfl.component.aggregation_functions.interface import AggregationFunction

n_classes = 2


class AggregateAdaboost(AggregationFunction):
    """Function for Adaboost.F aggregation"""

    def call(self, local_tensors, *_):
        tensors = [x.tensor for x in local_tensors]

        partial = []
        for tensor in tensors:
            partial.append(tensor[-1])
        norm = sum(partial)

        errors = []
        for i in range(len(tensors[0]) - 1):
            partial = []
            for tensor in tensors:
                partial.append(tensor[i])
            partial = np.array(partial)
            errors.append(partial.sum() / norm)

        errors = np.array(errors)
        best_model = np.argmin(errors)
        best_error = errors[best_model]
        alpha = np.log((1.0 - best_error) / (best_error + 1e-10)) + np.log(n_classes - 1)

        return np.array([alpha, best_model])
