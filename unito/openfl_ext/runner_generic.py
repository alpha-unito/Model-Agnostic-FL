"""GenericTaskRunner module."""

import pickle
import tqdm
import numpy

from logging import getLogger
from openfl.utilities import TensorKey
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from openfl.federated.task.runner import TaskRunner


class GenericTaskRunner(BaseEstimator, TaskRunner):
    """Federated Learning Task Runner Class."""

    def __init__(self, model, **kwargs):
        """
        Intialize.

        Args:
            data_loader: The data_loader object
            tensor_dict_split_fn_kwargs: (Default=None)
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)
        self.model = model

        # key word arguments for determining which parameters to hold out from
        # aggregation.
        # If set to none, an empty dict will be passed, currently resulting in
        # the defaults:
        # holdout_types=['non_float'] # all param np.arrays of this type will
        # be held out
        # holdout_tensor_names=[]     # params with these names will be held out
        # TODO: params are restored from protobufs as float32 numpy arrays, so
        # non-floats arrays and non-arrays are not currently supported for
        # passing to and from protobuf (and as a result for aggregation) - for
        # such params in current examples, aggregation does not make sense
        # anyway, but if this changes support should be added.
        self.set_logger()

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
        """Set data_loader object.

        Args:
            data_loader: data_loader object to set
        Returns:
            None
        """
        if data_loader.get_feature_shape() != \
                self.data_loader.get_feature_shape():
            raise ValueError(
                'The data_loader feature shape is not compatible with model.')

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

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """
        Set the model weights with a tensor dictionary:\
        {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables
                                  of the optimizer.

        Returns:
            None
        """
        raise NotImplementedError

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
