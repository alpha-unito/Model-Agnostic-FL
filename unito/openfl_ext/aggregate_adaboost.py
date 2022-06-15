import numpy as np
from openfl.component.aggregation_functions.interface import AggregationFunction

n_classes = 2


class AggregateAdaboost(AggregationFunction):
    """Function for Adaboost.F aggregation"""

    def call(self, local_tensors, *_):
        tensors = [x.tensor for x in local_tensors]
        sums = []
        for i in range(len(tensors)):
            partial = []
            for tensor in tensors:
                partial.append(tensor[i])
            sums.append(sum(partial) / len(tensors))

        c = np.argmin(sums)
        # e = (1 / (len(sums))) * sum([tensor[c] for tensor in tensors])
        # print(e)
        # a = np.log((1.0 - e) / (e + 1e-10)) + np.log(n_classes - 1)
        a = np.log((1.0 - sums[c]) / (sums[c] + 1e-10)) + np.log(n_classes - 1)

        return np.array([a, c])
