import numpy as np
from openfl.databases import TensorDB
from openfl.utilities import LocalTensor
from openfl.utilities import TensorKey


class TensorDB(TensorDB):

    def __init__(self, nn):
        super(TensorDB, self).__init__()
        self.nn = nn

    def get_aggregated_tensor(self, tensor_key, collaborator_weight_dict,
                              aggregation_function):
        """
        Determine whether all of the collaborator tensors are present for a given tensor key.

        Returns their weighted average.

        Args:
            tensor_key: The tensor key to be resolved. If origin 'agg_uuid' is
                        present, can be returned directly. Otherwise must
                        compute weighted average of all collaborators
            collaborator_weight_dict: List of collaborator names in federation
                                      and their respective weights
            aggregation_function: Call the underlying numpy aggregation
                                   function. Default is just the weighted
                                   average.
        Returns:
            weighted_nparray if all collaborator values are present
            None if not all values are present

        """
        if len(collaborator_weight_dict) != 0:
            assert np.abs(1.0 - sum(collaborator_weight_dict.values())) < 0.01, (
                f'Collaborator weights do not sum to 1.0: {collaborator_weight_dict}'
            )

        collaborator_names = collaborator_weight_dict.keys()
        agg_tensor_dict = {}

        # Check if the aggregated tensor is already present in TensorDB
        tensor_name, origin, fl_round, report, tags = tensor_key

        raw_df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_name)
                                & (self.tensor_db['origin'] == origin)
                                & (self.tensor_db['round'] == fl_round)
                                & (self.tensor_db['report'] == report)
                                & (self.tensor_db['tags'] == tags)]['nparray']
        if len(raw_df) > 0:
            return np.array(raw_df.iloc[0])  # TODO , {} this is strange

        for col in collaborator_names:
            if type(tags) == str:
                new_tags = tuple([tags] + [col])
            else:
                new_tags = tuple(list(tags) + [col])
            raw_df = self.tensor_db[
                (self.tensor_db['tensor_name'] == tensor_name)
                & (self.tensor_db['origin'] == origin)
                & (self.tensor_db['round'] == fl_round)
                & (self.tensor_db['report'] == report)
                & (self.tensor_db['tags'] == new_tags)]['nparray']
            if len(raw_df) == 0:
                tk = TensorKey(tensor_name, origin, report, fl_round, new_tags)
                print(f'No results for collaborator {col}, TensorKey={tk}')
                return None
            else:
                agg_tensor_dict[col] = raw_df.iloc[0]

        local_tensors = [LocalTensor(col_name=col_name,
                                     tensor=agg_tensor_dict[col_name],
                                     weight=collaborator_weight_dict[col_name])
                         for col_name in collaborator_names]

        db_iterator = self._iterate()
        agg_nparray = aggregation_function(local_tensors,
                                           db_iterator,
                                           tensor_name,
                                           fl_round,
                                           tags)
        self.cache_tensor({tensor_key: agg_nparray})

        if self.nn or 'metric' in tags:
            result = np.array(agg_nparray)
        else:
            result = agg_nparray

        return result

    def get_tensor_from_cache(self, tensor_key):
        """
        Perform a lookup of the tensor_key in the TensorDB.

        Returns the nparray if it is available
        Otherwise, it returns 'None'
        """
        tensor_name, origin, fl_round, report, tags = tensor_key

        # TODO come up with easy way to ignore compression
        df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_name)
                            & (self.tensor_db['origin'] == origin)
                            & (self.tensor_db['round'] == fl_round)
                            & (self.tensor_db['report'] == report)
                            & (self.tensor_db['tags'] == tags)]

        if len(df) == 0:
            return None

        # @TODO: this shuold differentiare between generic models and metrics
        if self.nn:
            result = np.array(df['nparray'].iloc[0])
        else:
            result = df['nparray'].iloc[0]

        return result

    # @TODO: this is also to be generalised
    def get_errors(self, round_number):
        df = self.tensor_db[(self.tensor_db['tensor_name'] == "errors")
                            & (self.tensor_db['round'] == round_number)]

        if len(df) == 0:
            return None
        return df["nparray"].to_numpy()
