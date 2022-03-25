import logging
import uuid

from openfl.transport import DirectorGRPCServer
from openfl.protocols import director_pb2
from openfl.protocols.utils import deconstruct_model_proto
from openfl.protocols.utils import construct_model_proto

from unito.openfl_ext.generic_pipeline import GenericPipeline

logger = logging.getLogger(__name__)


class DirectorGRPCServer(DirectorGRPCServer):

    async def SetNewExperiment(self, stream, context):  # NOQA:N802
        """Request to set new experiment."""
        logger.info(f'SetNewExperiment request has got {stream}')
        # TODO: add streaming reader
        data_file_path = self.root_dir / str(uuid.uuid4())
        with open(data_file_path, 'wb') as data_file:
            async for request in stream:
                if request.experiment_data.size == len(request.experiment_data.npbytes):
                    data_file.write(request.experiment_data.npbytes)
                else:
                    raise Exception('Bad request')

        tensor_dict = None
        if request.model_proto:
            # TODO the type of pipeline should not be fixed
            tensor_dict, _ = deconstruct_model_proto(request.model_proto, GenericPipeline())

        caller = self.get_caller(context)

        is_accepted = await self.director.set_new_experiment(
            experiment_name=request.name,
            sender_name=caller,
            tensor_dict=tensor_dict,
            collaborator_names=request.collaborator_names,
            experiment_archive_path=data_file_path
        )

        logger.info('Send response')
        return director_pb2.SetNewExperimentResponse(accepted=is_accepted)

    async def GetTrainedModel(self, request, context):  # NOQA:N802
        """RPC for retrieving trained models."""
        logger.info('Request GetTrainedModel has got!')

        if request.model_type == director_pb2.GetTrainedModelRequest.BEST_MODEL:
            model_type = 'best'
        elif request.model_type == director_pb2.GetTrainedModelRequest.LAST_MODEL:
            model_type = 'last'
        else:
            logger.error('Incorrect model type')
            return director_pb2.TrainedModelResponse()

        caller = self.get_caller(context)

        trained_model_dict = self.director.get_trained_model(
            experiment_name=request.experiment_name,
            caller=caller,
            model_type=model_type
        )

        if trained_model_dict is None:
            return director_pb2.TrainedModelResponse()

        model_proto = construct_model_proto(trained_model_dict, 0, GenericPipeline())

        return director_pb2.TrainedModelResponse(model_proto=model_proto)
