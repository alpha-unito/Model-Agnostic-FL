"""GenericPipeline module."""

from openfl.pipelines.pipeline import TransformationPipeline
from openfl.pipelines.pipeline import Float32NumpyArrayToBytes
from dill import dumps, loads

import numpy as np


class Float32NumpyArrayToBytes(Float32NumpyArrayToBytes):
    """Converts float32 Numpy array to Bytes array."""

    def __init__(self, nn):
        """Initialize."""
        self.lossy = False
        self.nn = nn

    def forward(self, data, **kwargs):
        """Forward pass.

        Args:
            data:
            **kwargs: Additional arguments to pass to the function

        Returns:
            data_bytes:
            metadata:
        """
        # TODO: Warn when this casting is being performed.
        if self.nn or isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            array_shape = data.shape
            # Better call it array_shape?
            metadata = {'int_list': list(array_shape)}
            data_bytes = data.tobytes(order='C')
        else:
            data_bytes = dumps(data)
            metadata = {'model': True}

        return data_bytes, metadata

    def backward(self, data, metadata, **kwargs):
        """Backward pass.

        Args:
            data:
            metadata:

        Returns:
            Numpy Array

        """
        if not metadata['model']:
            array_shape = tuple(metadata['int_list'])
            flat_array = np.frombuffer(data, dtype=np.float32)
            # For integer parameters we probably should unpack arrays
            # with shape (1,)
            result = np.reshape(flat_array, newshape=array_shape, order='C')
        else:
            result = loads(data)

        return result


class TransformationPipeline(TransformationPipeline):
    """The generic data pipeline without any compression"""

    def __init__(self, nn, transformers, **kwargs):
        super(TransformationPipeline, self).__init__(transformers, **kwargs)
        self.nn = nn

    def forward(self, data, **kwargs):
        """Forward pass of pipeline data transformer.

        Args:
            data: Data to transform
            **kwargs: Additional parameters to pass to the function

        Returns:
            data:
            transformer_metadata:

        """
        transformer_metadata = []

        # dataformat::numpy::float.32
        # model proto:: a collection of tensor_dict proto
        # protobuff::-> a layer of weights
        # input::tensor_dict:{"layer1":np.array(float32, [128,3,28,28]),
        # "layer2": np.array()}
        # output::meta data::numpy::int array
        # (data, transformer_metadata)::(float32, dictionary o
        # key+float32 vlues)
        # input:: numpy_data (float32)
        # input:: (data(bytes), transformer_metadata_list::a list of dictionary
        # from int to float)
        if self.nn or isinstance(data, np.ndarray):
            data = data.copy()

        for transformer in self.transformers:
            data, metadata = transformer.forward(data, **kwargs)
            transformer_metadata.append(metadata)
        return data, transformer_metadata

    def backward(self, data, transformer_metadata, **kwargs):
        """Backward pass of pipeline data transformer.

        Args:
            data: Data to transform
            transformer_metadata:
            **kwargs: Additional parameters to pass to the function

        Returns:
            data:

        """
        for transformer in self.transformers[::-1]:
            data = transformer.backward(
                data=data, metadata=transformer_metadata.pop(), **kwargs)
        return data

    def is_lossy(self):
        """If any of the transformers are lossy, then the pipeline is lossy."""
        return any([transformer.lossy for transformer in self.transformers])
