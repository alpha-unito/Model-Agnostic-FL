"""GenericTaskRunner module."""

import pickle
import tqdm
import numpy
from logging import getLogger
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey

from openfl.federated.task.task_runner import CoreTaskRunner


class GenericTaskRunner(BaseEstimator, CoreTaskRunner):
    """Federated Learning Task Runner Class."""

    def adapt_tasks(self):
        """
        Prepare tasks for the collaborator.

        Using functions from a task provider (deserialized interface object) and
        registered task contracts prepares callable tasks to be invoked by the collaborator.

        Preparing includes conditional model rebuilding and filling output dicts
        with tensors for aggregation and storing in local DB.

        There is an assumption that any training task accepts optimizer as one
        of the arguments, thus the model should be aggregated after such tasks.
        """

        def task_binder(task_name, callable_task):
            def collaborator_adapted_task(col_name, round_num, input_tensor_dict, **kwargs):
                task_contract = self.task_provider.task_contract[task_name]
                # Validation flag can be [False, '_local', '_agg']
                validation_flag = True if task_contract['optimizer'] is None else False
                task_settings = self.task_provider.task_settings[task_name]

                device = kwargs.get('device', 'cpu')

                self.rebuild_model(input_tensor_dict, validation=validation_flag, device=device)
                task_kwargs = {}
                if validation_flag:
                    loader = self.data_loader.get_valid_loader()
                    if kwargs['apply'] == 'local':
                        validation_flag = '_local'
                    else:
                        validation_flag = '_agg'
                else:
                    loader = self.data_loader.get_train_loader()
                    # If train task we also pass optimizer
                    task_kwargs[task_contract['optimizer']] = self.optimizer

                for en_name, entity in zip(['model', 'data_loader', 'device'],
                                           [self.model, loader, device]):
                    task_kwargs[task_contract[en_name]] = entity

                # Add task settings to the keyword arguments
                task_kwargs.update(task_settings)

                # Here is the training metod call
                metric_dict = callable_task(**task_kwargs)

                return self._prepare_tensorkeys_for_agggregation(
                    metric_dict, validation_flag, col_name, round_num)

            return collaborator_adapted_task

        for task_name, callable_task in self.task_provider.task_registry.items():
            self.TASK_REGISTRY[task_name] = task_binder(task_name, callable_task)

    def _prepare_tensorkeys_for_agggregation(self, metric_dict, validation_flag,
                                             col_name, round_num, nn=False):  # TODO This should be in the plan file
        """
        Prepare tensorkeys for aggregation.

        Returns (global_tensor_dict, local_tensor_dict)
        """
        global_tensor_dict, local_tensor_dict = {}, {}
        origin = col_name
        if not validation_flag:
            # Output metric tensors (scalar)
            tags = ('trained',)

            # output model tensors (Doesn't include TensorKey)
            output_model_dict = self.get_tensor_dict(with_opt_vars=True)
            global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                self.logger, output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

            # Create global tensorkeys
            global_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags):
                    nparray for tensor_name, nparray in global_model_dict.items()}
            # Create tensorkeys that should stay local
            local_tensorkey_model_dict = {
                TensorKey(tensor_name, origin, round_num, False, tags):
                    nparray for tensor_name, nparray in local_model_dict.items()}
            # The train/validate aggregated function of the next
            # round will look for the updated model parameters.
            # This ensures they will be resolved locally
            next_local_tensorkey_model_dict = {TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)): nparray
                                               for tensor_name, nparray in local_model_dict.items()}

            global_tensor_dict = global_tensorkey_model_dict
            local_tensor_dict = {**local_tensorkey_model_dict, **next_local_tensorkey_model_dict}

            # Update the required tensors if they need to be
            # pulled from the aggregator
            # TODO this logic can break if different collaborators
            #  have different roles between rounds.
            # For example, if a collaborator only performs validation
            # in the first round but training
            # in the second, it has no way of knowing the optimizer
            # state tensor names to request from the aggregator
            # because these are only created after training occurs.
            # A work around could involve doing a single epoch of training
            # on random data to get the optimizer names,
            # and then throwing away the model.
            if self.opt_treatment == 'CONTINUE_GLOBAL':
                self.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
            self.training_round_completed = True

        else:
            suffix = 'validate' + validation_flag
            tags = (suffix,)
        tags = ('metric', *tags)
        metric_dict = {
            TensorKey(metric, origin, round_num, True, tags):
                np.array(value) for metric, value in metric_dict.items()
        }
        global_tensor_dict = {**global_tensor_dict, **metric_dict}

        return global_tensor_dict, local_tensor_dict

    def set_logger(self):
        """Set up the log object."""
        self.logger = getLogger(__name__)

    def get_data_loader(self):
        """
        Get the data_loader object.

        Serves up batches and provides info regarding data_loader.

        Returns:
            data_loader object
        """
        return self.data_loader

    def set_data_loader(self, data_loader):
        """Register a data loader initialized with local data path."""
        self.data_loader = data_loader

    def get_train_data_size(self):
        """
        Get the number of training examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of training examples.
        """
        return self.data_loader.get_train_data_size()

    def get_valid_data_size(self):
        """
        Get the number of examples.

        It will be used for weighted averaging in aggregation.

        Returns:
            int: The number of validation examples.
        """
        return self.data_loader.get_valid_data_size()

    def train(self, col_name, round_num, input_tensor_dict, num_batches=None, use_tqdm=False, **kwargs):
        """Train batches.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            num_batches:         The number of batches to train on before
                                 returning
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """

        loader = self.data_loader.get_train_loader(num_batches)
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="train epoch")
        metric = None
        for data, target in loader:
            self.model = self.model.fit(data, target)
            pred = self.model.predict(data)
            metric = accuracy_score(target, pred, normalize=True)
        tags = ('trained',)
        origin = col_name
        output_metric_dict = {
            TensorKey(
                'acc', origin, round_num, True, ('metric',)
            ): numpy.array(metric)
        }

        # output model tensors (Doesn't include TensorKey)
        output_model = self.model

        global_tensorkey_model_dict = {
            TensorKey('model', origin, round_num, False, tags):
                output_model
        }
        local_tensorkey_model_dict = {
            TensorKey('model', origin, round_num, False, tags):
                output_model
        }

        next_local_tensorkey_model_dict = {
            TensorKey('model', origin, round_num + 1, False, ('model',)):
                output_model
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        return global_tensor_dict, local_tensor_dict

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        if input_tensor_dict["model"] is not None:
            self.model = input_tensor_dict["model"]
        origin = col_name
        total_samples = 0

        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)

        metric = None
        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")
        try:
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                pred = self.model.predict(data)
                metric = accuracy_score(target, pred, normalize=True)
        except NotFittedError as _:
            metric = 0.
        output_tensor_dict = {
            TensorKey('acc', origin, round_num, True, tags):
                numpy.array(metric)
        }

        return output_tensor_dict, {}

    def set_framework_adapter(self, framework_adapter):
        """
        Set framework adapter.

        Setting a framework adapter allows first extraction of the weigths
        of the model with the purpose to make a list of parameters to be aggregated.
        """
        self.framework_adapter = framework_adapter
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            aggregate_optimizer_parameters = True
        else:
            aggregate_optimizer_parameters = False
        self.initialize_tensorkeys_for_functions(with_opt_vars=aggregate_optimizer_parameters)

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False, nn=False):
        """Set the required tensors for all publicly accessible task methods.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function.

        Args:
            None

        Returns:
            None
        """
        if nn:
            # TODO: Framework adapters should have separate methods for dealing with optimizer
            # Set model dict for validation tasks
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )
            # Now set model dict for training tasks
            if with_opt_vars:
                output_model_dict = self.get_tensor_dict(with_opt_vars=True)
                global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
                    self.logger, output_model_dict,
                    **self.tensor_dict_split_fn_kwargs
                )
            else:
                global_model_dict = global_model_dict_val
                local_model_dict = local_model_dict_val

            self.required_tensorkeys_for_function['global_model_dict'] = global_model_dict
            self.required_tensorkeys_for_function['local_model_dict'] = local_model_dict
            self.required_tensorkeys_for_function['global_model_dict_val'] = global_model_dict_val
            self.required_tensorkeys_for_function['local_model_dict_val'] = local_model_dict_val
        else:
            model = self.get_tensor_dict(with_opt_vars=False)

            self.required_tensorkeys_for_function['global_model_dict'] = model
            self.required_tensorkeys_for_function['local_model_dict'] = model
            self.required_tensorkeys_for_function['global_model_dict_val'] = model
            self.required_tensorkeys_for_function['local_model_dict_val'] = model

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        When running a task, a map of named tensorkeys \
            must be provided to the function as dependencies.

        Returns:
            list: (TensorKey(tensor_name, origin, round_number))
        """
        raise NotImplementedError

    def get_tensor_dict(self, with_opt_vars):
        """
        Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
                                  of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}
        """

        return pickle.dumps(self.model)

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        raise NotImplementedError

    def initialize_globals(self):
        """
        Initialize all global variables.

        Returns:
            None
        """
        raise NotImplementedError

    def load_native(self, filepath, **kwargs):
        """
        Load model state from a filepath in ML-framework "native" format, \
            e.g. PyTorch pickled models.

        May load from multiple files. Other filepaths may be derived from the
        passed filepath, or they may be in the kwargs.

        Args:
            filepath (string): Path to frame-work specific file to load. For
            frameworks that use multiple files, this string must be used to
            derive the other filepaths.
            kwargs           : For future-proofing

        Returns:
            None
        """
        raise NotImplementedError

    def save_native(self, filepath, **kwargs):
        """
        Save model state in ML-framework "native" format, e.g. PyTorch pickled models.

        May save one file or multiple files, depending on the framework.

        Args:
            filepath (string): If framework stores a single file, this should
                               be a single file path.
            Frameworks that store multiple files may need to derive the other
            paths from this path.
            kwargs           : For future-proofing

        Returns:
            None
        """
        raise NotImplementedError
