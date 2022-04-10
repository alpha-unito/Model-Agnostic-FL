import logging
import time
from pathlib import Path
from click import echo

from openfl.utilities.workspace import ExperimentWorkspace
from openfl.component.envoy.envoy import Envoy

from unito.openfl_ext.plan import Plan

logger = logging.getLogger(__name__)

DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5


class Envoy(Envoy):

    def start(self):
        """Start the envoy_1."""
        try:
            is_accepted = self.director_client.report_shard_info(
                shard_descriptor=self.shard_descriptor,
                cuda_devices=self.cuda_devices)
        except Exception as exc:
            logger.exception(f'Failed to report shard info: {exc}')
        else:
            if is_accepted:
                # Shard accepted for participation in the federation
                logger.info('Shard accepted')
                self._health_check_future = self.executor.submit(self.send_health_check)
                self.run()
            else:
                # Shut down
                logger.error('Report shard info was not accepted')

    def run(self):
        """Run of the envoy_1 working cycle."""
        while True:
            try:
                # Workspace import should not be done by gRPC client!
                experiment_name = self.director_client.wait_experiment()
                data_stream = self.director_client.get_experiment_data(experiment_name)
            except Exception as exc:
                logger.exception(f'Failed to get experiment: {exc}')
                time.sleep(DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            data_file_path = self._save_data_stream_to_file(data_stream)
            self.is_experiment_running = True
            try:
                with ExperimentWorkspace(
                        experiment_name, data_file_path, is_install_requirements=True
                ):
                    self._run_collaborator()
            except Exception as exc:
                logger.exception(f'Collaborator failed with error: {exc}:')
            finally:
                self.is_experiment_running = False

    def _run_collaborator(self, plan='plan/plan.yaml'):
        """Run the collaborator for the experiment running."""
        plan = Plan.parse(plan_config_path=Path(plan))

        # TODO: Need to restructure data loader config file loader
        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, self.root_certificate, self.private_key,
                                    self.certificate, shard_descriptor=self.shard_descriptor)
        col.set_available_devices(cuda=self.cuda_devices)
        col.run()
