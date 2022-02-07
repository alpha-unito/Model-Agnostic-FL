from openfl.pipelines.tensor_codec import TensorCodec
from openfl.utilities import TensorKey


class TensorCodec(TensorCodec):
    """TensorCodec is responsible for the following.

        1. Tracking the compression/decompression related dependencies of a given tensor
        2. Acting as a TensorKey aware wrapper for the compression_pipeline functionality
        """

    def __init__(self, compression_pipeline):
        super(TensorCodec, self).__init__(compression_pipeline)

    def compress(self, tensor_key, data, require_lossless=False, **kwargs):
        """
        Function-wrapper around the tensor_pipeline.forward function.

        It also keeps track of the tensorkeys associated with the compressed nparray

        Args:
            tensor_key:             TensorKey is provided to verify it should
                                    be compressed, and new TensorKeys returned
                                    will be derivatives of the existing
                                    tensor_name

            data:                   (uncompressed) numpy array associated with
                                    the tensor_key

            require_lossless:       boolean. Does tensor require
                                    compression

        Returns:
            compressed_tensor_key:  Tensorkey corresponding to the decompressed
                                    tensor

            compressed_nparray:     The compressed tensor

            metadata:               metadata associated with compressed tensor

        """
        if require_lossless:
            compressed_nparray, metadata = self.lossless_pipeline.forward(
                data, **kwargs)
        else:
            compressed_nparray, metadata = self.compression_pipeline.forward(
                data, **kwargs)
        # Define the compressed tensorkey that should be
        # returned ('trained.delta'->'trained.delta.lossy_compressed')
        tensor_name, origin, round_number, report, tags = tensor_key
        if not self.compression_pipeline.is_lossy() or require_lossless:
            new_tags = tuple(list(tags) + ['compressed'])
        else:
            new_tags = tuple(list(tags) + ['lossy_compressed'])
        compressed_tensor_key = TensorKey(
            tensor_name, origin, round_number, report, new_tags)
        return compressed_tensor_key, compressed_nparray, metadata

    def decompress(self, tensor_key, data, transformer_metadata,
                   require_lossless=False, **kwargs):
        """
        Function-wrapper around the tensor_pipeline.backward function.

        It also keeps track of the tensorkeys associated with the decompressed nparray

        Args:
            tensor_key:             TensorKey is provided to verify it should
                                    be decompressed, and new TensorKeys
                                    returned will be derivatives of the
                                    existing tensor_name

            data:                   (compressed) numpy array associated with
                                    the tensor_key

            transformer_metadata:   metadata associated with the compressed
                                    tensor

            require_lossless:       boolean, does data require lossless
                                    decompression

        Returns:
            decompressed_tensor_key:    Tensorkey corresponding to the
                                        decompressed tensor

            decompressed_nparray:       The decompressed tensor

        """
        tensor_name, origin, round_number, report, tags = tensor_key

        assert (len(transformer_metadata) > 0), (
            'metadata must be included for decompression')
        assert (('compressed' in tags) or ('lossy_compressed' in tags)), (
            'Cannot decompress an uncompressed tensor')
        if require_lossless:
            assert ('compressed' in tags), (
                'Cannot losslessly decompress lossy tensor')

        if require_lossless or 'compressed' in tags:
            decompressed_nparray = self.lossless_pipeline.backward(
                data, transformer_metadata, **kwargs)
        else:
            decompressed_nparray = self.compression_pipeline.backward(
                data, transformer_metadata, **kwargs)
        # Define the decompressed tensorkey that should be returned
        if 'lossy_compressed' in tags:
            lc_idx = tags.index('lossy_compressed')
            new_tags = list(tags)
            new_tags[lc_idx] = 'lossy_decompressed'
            decompressed_tensor_key = TensorKey(
                tensor_name, origin, round_number, report, tuple(new_tags))
        elif 'compressed' in tags:
            # 'compressed' == lossless compression; no need for
            # compression related tag after decompression
            new_tags = list(tags)
            new_tags.remove('compressed')
            decompressed_tensor_key = TensorKey(
                tensor_name, origin, round_number, report, tuple(new_tags))
        else:
            raise NotImplementedError(
                'Decompression is only supported on compressed data')

        return decompressed_tensor_key, decompressed_nparray
