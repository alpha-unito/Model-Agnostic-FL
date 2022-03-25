"""GenericPipeline module."""

from unito.openfl_ext.pipeline import TransformationPipeline, Float32NumpyArrayToBytes


class GenericPipeline(TransformationPipeline):
    """The data pipeline without any compression."""

    def __init__(self, nn=False, **kwargs):
        """Initialize."""
        super(GenericPipeline, self).__init__(nn=nn, transformers=[Float32NumpyArrayToBytes(nn)], **kwargs)
        self.nn = nn

    def is_nn(self):
        return self.nn
