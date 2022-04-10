from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import logging
import asyncio

from openfl.component.director.experiment import Experiment
from openfl.utilities.workspace import ExperimentWorkspace
from openfl.transport import AggregatorGRPCServer

from unito.openfl_ext.plan import Plan

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = 'pending'
    FINISHED = 'finished'
    IN_PROGRESS = 'in_progress'
    FAILED = 'failed'


class Experiment(Experiment):

    def __init__(
            self, *,
            name: str,
            archive_path: Union[Path, str],
            collaborators: List[str],
            sender: str,
            init_tensor_dict: dict,
            plan_path: Union[Path, str] = 'plan/plan.yaml',
            users: Iterable[str] = None,
    ) -> None:
        """Initialize an experiment object."""
        self.name = name
        if isinstance(archive_path, str):
            archive_path = Path(archive_path)
        self.archive_path = archive_path
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict = init_tensor_dict
        if isinstance(plan_path, str):
            plan_path = Path(plan_path)
        self.plan_path = plan_path
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.aggregator = None

    async def start(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ):
        """Run experiment."""
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f'New experiment {self.name} for '
                        f'collaborators {self.collaborators}')

            with ExperimentWorkspace(self.name, self.archive_path):
                aggregator_grpc_server = self._create_aggregator_grpc_server(
                    tls=tls,
                    root_certificate=root_certificate,
                    private_key=private_key,
                    certificate=certificate,
                )
                self.aggregator = aggregator_grpc_server.aggregator
                await self._run_aggregator_grpc_server(
                    aggregator_grpc_server=aggregator_grpc_server,
                )
            self.status = Status.FINISHED
            logger.info(f'Experiment "{self.name}" was finished successfully.')
        except Exception as e:
            self.status = Status.FAILED
            logger.error(f'Experiment "{self.name}" was failed with error: {e}.')

    def _create_aggregator_grpc_server(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ) -> AggregatorGRPCServer:
        plan = Plan.parse(plan_config_path=Path(self.plan_path))
        plan.authorized_cols = list(self.collaborators)

        logger.info('🧿 Starting the Aggregator Service.')
        aggregator_grpc_server = plan.interactive_api_get_server(
            tensor_dict=self.init_tensor_dict,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )

        return aggregator_grpc_server

    @staticmethod
    async def _run_aggregator_grpc_server(aggregator_grpc_server: AggregatorGRPCServer) -> None:
        """Run aggregator."""
        logger.info('🧿 Starting the Aggregator Service.')
        grpc_server = aggregator_grpc_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not aggregator_grpc_server.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            aggregator_grpc_server.aggregator.tensor_db.clean_up(0)  # TODO this is gonna crash if no envoy_1 is active
