import numpy as np
from openfl.component.aggregation_functions.interface import AggregationFunction

n_classes = 2


class AggregateAdaboost(AggregationFunction):
    """Function for Adaboost.F aggregation"""

    # @TODO: are the tensors ordered by client rank?
    def call(self, local_tensors, *_):
        tensors = [x.tensor for x in local_tensors]
        sums = []
        for tensor in tensors:
            sums.append(sum(tensor))
        c = np.argmin(sums)
        e = (1 / (len(sums))) * sum([tensor[c] for tensor in tensors])
        a = np.log((1.0 - e) / (e + 1e-10)) + np.log(n_classes - 1)

        print(tensors)
        print(c)
        print(e)
        print(a)

        return np.array([a, c])
