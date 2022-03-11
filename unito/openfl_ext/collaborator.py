from openfl.component.collaborator import Collaborator
from openfl.component.collaborator.collaborator import OptTreatment
from openfl.component.collaborator.collaborator import DevicePolicy
from openfl.pipelines import NoCompressionPipeline
from openfl.utilities import TensorKey
from logging import getLogger
from time import sleep
from openfl.protocols import utils

from .tensor_codec import TensorCodec
from .tensor_db import TensorDB


class Collaborator(Collaborator):
    r"""The Collaborator object class.

        Args:
            collaborator_name (string): The common name for the collaborator
            aggregator_uuid: The unique id for the client
            federation_uuid: The unique id for the federation
            model: The model
            opt_treatment* (string): The optimizer state treatment (Defaults to
                "CONTINUE_GLOBAL", which is aggreagated state from previous round.)

            compression_pipeline: The compression pipeline (Defaults to None)

            num_batches_per_round (int): Number of batches per round
                                         (Defaults to None)

            delta_updates* (bool): True = Only model delta gets sent.
                                   False = Whole model gets sent to collaborator.
                                   Defaults to False.

            single_col_cert_common_name: (Defaults to None)

        Note:
            \* - Plan setting.
        """

    def __init__(self,
                 collaborator_name,
                 aggregator_uuid,
                 federation_uuid,
                 client,
                 task_runner,
                 task_config,
                 opt_treatment='RESET',
                 device_assignment_policy='CPU_ONLY',
                 delta_updates=False,
                 compression_pipeline=None,
                 db_store_rounds=1,
                 nn=False,
                 **kwargs):
        """Initialize."""
        self.single_col_cert_common_name = None

        self.nn = nn

        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = ''  # for protobuf compatibility
        # we would really want this as an object

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.tensor_db = TensorDB(self.nn)
        self.db_store_rounds = db_store_rounds

        self.task_runner = task_runner
        self.delta_updates = delta_updates

        self.client = client

        self.task_config = task_config

        self.logger = getLogger(__name__)

        # RESET/CONTINUE_LOCAL/CONTINUE_GLOBAL
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error(f'Unknown opt_treatment: {opt_treatment.name}.')
            raise NotImplementedError(f'Unknown opt_treatment: {opt_treatment}.')

        if hasattr(DevicePolicy, device_assignment_policy):
            self.device_assignment_policy = DevicePolicy[device_assignment_policy]
        else:
            self.logger.error('Unknown device_assignment_policy: '
                              f'{device_assignment_policy.name}.')
            raise NotImplementedError(
                f'Unknown device_assignment_policy: {device_assignment_policy}.'
            )

        self.task_runner.set_optimizer_treatment(self.opt_treatment.name)

    def run_simulation(self):
        """
        Specific function for the simulation.

        After the tasks have
        been performed for a roundquit, and then the collaborator object will
        be reinitialized after the next round
        """
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                self.logger.info('End of Federation reached. Exiting...')
                break
            elif sleep_time > 0:
                sleep(sleep_time)  # some sleep function
            else:
                self.logger.info(f'Received the following tasks: {tasks}')
                for task in tasks:
                    self.do_task(task, round_number)
                self.logger.info(f'All tasks completed on {self.collaborator_name} '
                                 f'for round {round_number}...')
                break

    def do_task(self, task, round_number):
        """Do the specified task."""
        # map this task to an actual function name and kwargs
        func_name = self.task_config[task]['function']
        kwargs = self.task_config[task]['kwargs']

        if self.nn:
            # this would return a list of what tensors we require as TensorKeys
            required_tensorkeys_relative = self.task_runner.get_required_tensorkeys_for_function(
                func_name,
                **kwargs
            )

            # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL,
            # round_offset)
            # so we need to update these keys to their "absolute values"
            required_tensorkeys = []
            for tname, origin, rnd_num, report, tags in required_tensorkeys_relative:
                if origin == 'GLOBAL':
                    origin = self.aggregator_uuid
                else:
                    origin = self.collaborator_name

                # rnd_num is the relative round. So if rnd_num is -1, get the
                # tensor from the previous round
                required_tensorkeys.append(
                    TensorKey(tname, origin, rnd_num + round_number, report, tags)
                )

            # print('Required tensorkeys = {}'.format(
            # [tk[0] for tk in required_tensorkeys]))
            input_tensor_dict = self.get_numpy_dict_for_tensorkeys(
                required_tensorkeys
            )
        else:
            input_tensor_dict = {"model": None}

        # now we have whatever the model needs to do the task
        if hasattr(self.task_runner, 'TASK_REGISTRY'):
            # New interactive python API
            # New `Core` TaskRunner contains registry of tasks
            func = self.task_runner.TASK_REGISTRY[func_name]
            self.logger.info('Using Interactive Python API')

            # So far 'kwargs' contained parameters read from the plan
            # those are parameters that the eperiment owner registered for
            # the task.
            # There is another set of parameters that created on the
            # collaborator side, for instance, local processing unit identifier:s
            if (self.device_assignment_policy is DevicePolicy.CUDA_PREFERRED
                    and len(self.cuda_devices) > 0):
                kwargs['device'] = f'cuda:{self.cuda_devices[0]}'
            else:
                kwargs['device'] = 'cpu'
        else:
            # TaskRunner subclassing API
            # Tasks are defined as methods of TaskRunner
            func = getattr(self.task_runner, func_name)
            self.logger.info('Using TaskRunner subclassing API')

        global_output_tensor_dict, local_output_tensor_dict = func(
            col_name=self.collaborator_name,
            round_num=round_number,
            input_tensor_dict=input_tensor_dict,
            **kwargs)

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        self.send_task_results(global_output_tensor_dict, round_number, task)

    def send_task_results(self, tensor_dict, round_number, task_name):
        """Send task results to the aggregator."""
        named_tensors = [
            self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()
        ]

        # for general tasks, there may be no notion of data size to send.
        # But that raises the question how to properly aggregate results.

        data_size = -1

        if 'train' in task_name:
            data_size = self.task_runner.get_train_data_size()

        if 'valid' in task_name:
            data_size = self.task_runner.get_valid_data_size()

        self.logger.debug(f'{task_name} data size = {data_size}')

        for tensor in tensor_dict:
            tensor_name, origin, fl_round, report, tags = tensor

            if report:
                self.logger.metric(
                    f'Round {round_number}, collaborator {self.collaborator_name} '
                    f'is sending metric for task {task_name}:'
                    f' {tensor_name}\t{tensor_dict[tensor]}')

        self.client.send_local_task_results(
            self.collaborator_name, round_number, task_name, data_size, named_tensors)

    def nparray_to_named_tensor(self, tensor_key, nparray):
        """
        Construct the NamedTensor Protobuf.

        Includes logic to create delta, compress tensors with the TensorCodec, etc.
        """
        # if we have an aggregated tensor, we can make a delta
        tensor_name, origin, round_number, report, tags = tensor_key
        if 'trained' in tags and self.delta_updates:
            # Should get the pretrained model to create the delta. If training
            # has happened,
            # Model should already be stored in the TensorDB
            model_nparray = self.tensor_db.get_tensor_from_cache(
                TensorKey(
                    tensor_name,
                    origin,
                    round_number,
                    report,
                    ('model',)
                )
            )

            # The original model will not be present for the optimizer on the
            # first round.
            if model_nparray is not None:
                delta_tensor_key, delta_nparray = self.tensor_codec.generate_delta(
                    tensor_key,
                    nparray,
                    model_nparray
                )
                delta_comp_tensor_key, delta_comp_nparray, metadata = self.tensor_codec.compress(
                    delta_tensor_key,
                    delta_nparray,
                    self.nn
                )

                named_tensor = utils.construct_named_tensor(
                    delta_comp_tensor_key,
                    delta_comp_nparray,
                    metadata,
                    lossless=False
                )
                return named_tensor

        # Assume every other tensor requires lossless compression
        compressed_tensor_key, compressed_nparray, metadata = self.tensor_codec.compress(
            tensor_key,
            nparray,
            require_lossless=True
        )
        named_tensor = utils.construct_named_tensor(
            compressed_tensor_key,
            compressed_nparray,
            metadata,
            lossless=True
        )

        return named_tensor
