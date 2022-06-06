import numpy as np
from openfl.component.aggregation_functions.interface import AggregationFunction


class AggregateAdaboost(AggregationFunction):
    """Function for Adaboost.F aggregation"""

    # @TODO: are the tensors ordered by client rank?
    def call(self, local_tensors, *_):
        tensors = [x.tensor for x in local_tensors]
        sums = []
        for tensor in tensors:
            sums.append(sum(tensor))
        c = np.argmin(sums)
        e = 1 / (len(sums)) * sum([tensor[c] for tensor in tensors])
        a = np.log((1 - e) / e)

        return np.array([a, c])
