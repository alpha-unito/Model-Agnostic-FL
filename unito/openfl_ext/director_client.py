import logging
from openfl.transport.grpc.director_client import DirectorClient
from openfl.pipelines import NoCompressionPipeline

from .protocols.utils import construct_model_proto
from .generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


class DirectorClient(DirectorClient):

    def set_new_experiment(self, name, col_names, arch_path,
                           initial_tensor_dict=None):
        """Send the new experiment to director to launch."""
        logger.info('SetNewExperiment')
        if initial_tensor_dict:
            model_proto = construct_model_proto(initial_tensor_dict, 0, NoCompressionPipeline())
            experiment_info_gen = self._get_experiment_info(
                arch_path=arch_path,
                name=name,
                col_names=col_names,
                model_proto=model_proto,
            )
            resp = self.stub.SetNewExperiment(experiment_info_gen)
        else:
            model_proto = construct_model_proto({'model': None}, 0, GenericPipeline())
            experiment_info_gen = self._get_experiment_info(
                arch_path=arch_path,
                name=name,
                col_names=col_names,
                model_proto=model_proto,
            )
            resp = self.stub.SetNewExperiment(experiment_info_gen)

        return resp
