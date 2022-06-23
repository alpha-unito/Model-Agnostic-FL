from openfl.protocols import SynchResponse
from openfl.protocols import TensorResponse
from openfl.transport import AggregatorGRPCServer


class AggregatorGRPCServer(AggregatorGRPCServer):

    def GetTensor(self, request, context):
        """
        Request a job from aggregator.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        tensor_name = request.tensor_name
        require_lossless = request.require_lossless
        round_number = request.round_number
        report = request.report
        tags = request.tags

        named_tensor = self.aggregator.get_tensor(
            collaborator_name, tensor_name, round_number, report, tags, require_lossless)

        return TensorResponse(header=self.get_header(collaborator_name),
                              round_number=round_number,
                              tensor=named_tensor)

    def GetSynch(self, request, context):  # NOQA:N802
        """
        Request synchronization for a task.

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        self.check_request(request)
        collaborator_name = request.header.sender
        task_name = request.task_name
        round_number = request.round_number
        is_completed = self.aggregator.synch(task_name, round_number)

        return SynchResponse(
            header=self.get_header(collaborator_name),
            is_completed=is_completed
        )
