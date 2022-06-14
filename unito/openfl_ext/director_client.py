import logging

from openfl.protocols import director_pb2
from openfl.protocols.utils import construct_model_proto
from openfl.protocols.utils import deconstruct_model_proto
from openfl.transport.grpc.director_client import DirectorClient

from unito.openfl_ext.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


class DirectorClient(DirectorClient):

    def set_new_experiment(self, name, col_names, arch_path,
                           initial_tensor_dict=None):
        """Send the new experiment to director to launch."""
        logger.info('SetNewExperiment')
        model_proto = construct_model_proto(initial_tensor_dict, 0, GenericPipeline())
        experiment_info_gen = self._get_experiment_info(
            arch_path=arch_path,
            name=name,
            col_names=col_names,
            model_proto=model_proto,
        )
        resp = self.stub.SetNewExperiment(experiment_info_gen)
        return resp

    def _get_trained_model(self, experiment_name, model_type):
        """Get trained model RPC."""
        get_model_request = director_pb2.GetTrainedModelRequest(
            experiment_name=experiment_name,
            model_type=model_type,
        )
        model_proto_response = self.stub.GetTrainedModel(get_model_request)
        tensor_dict, _ = deconstruct_model_proto(
            model_proto_response.model_proto,
            GenericPipeline(),
        )
        return tensor_dict

    def get_best_model(self, experiment_name):
        """Get best model method."""
        model_type = director_pb2.GetTrainedModelRequest.BEST_MODEL
        return self._get_trained_model(experiment_name, model_type)

    def get_last_model(self, experiment_name):
        """Get last model method."""
        model_type = director_pb2.GetTrainedModelRequest.LAST_MODEL
        return self._get_trained_model(experiment_name, model_type)
