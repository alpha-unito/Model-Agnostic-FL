from logging import getLogger
from pathlib import Path

from openfl.federated.plan import Plan
from openfl.interface.cli_helper import WORKSPACE
from openfl.transport import AggregatorGRPCServer
from yaml import dump

SETTINGS = 'settings'
TEMPLATE = 'template'
DEFAULTS = 'defaults'
AUTO = 'auto'


class Plan(Plan):
    """Federated Learning plan."""

    logger = getLogger(__name__)

    @staticmethod
    def parse(plan_config_path: Path, cols_config_path: Path = None,
              data_config_path: Path = None, resolve=True):
        """
        Parse the Federated Learning plan.

        Args:
            plan_config_path (string): The filepath to the federated learning
                                       plan
            cols_config_path (string): The filepath to the federation
                                       collaborator list [optional]
            data_config_path (string): The filepath to the federation
                                       collaborator data configuration
                                       [optional]
        Returns:
            A federated learning plan object
        """
        try:

            plan = Plan()
            plan.config = Plan.load(plan_config_path)  # load plan configuration
            plan.name = plan_config_path.name
            plan.files = [plan_config_path]  # collect all the plan files

            # ensure 'settings' appears in each top-level section
            for section in plan.config.keys():

                if plan.config[section].get(SETTINGS) is None:
                    plan.config[section][SETTINGS] = {}

            # walk the top level keys and load 'defaults' in sorted order
            for section in sorted(plan.config.keys()):
                defaults = plan.config[section].pop(DEFAULTS, None)

                if defaults is not None:
                    defaults = WORKSPACE / 'workspace' / defaults

                    plan.files.append(defaults)

                    if resolve:
                        Plan.logger.info(
                            f'Loading DEFAULTS for section [red]{section}[/] '
                            f'from file [red]{defaults}[/].',
                            extra={'markup': True})

                    defaults = Plan.load(Path(defaults))

                    if SETTINGS in defaults:
                        # override defaults with section settings
                        defaults[SETTINGS].update(
                            plan.config[section][SETTINGS])
                        plan.config[section][SETTINGS] = defaults[SETTINGS]

                    defaults.update(plan.config[section])

                    plan.config[section] = defaults

            plan.authorized_cols = Plan.load(cols_config_path).get(
                'collaborators', []
            )

            # TODO: Does this need to be a YAML file? Probably want to use key
            #  value as the plan hash
            plan.cols_data_paths = {}
            if data_config_path is not None:
                data_config = open(data_config_path, 'r')
                for line in data_config:
                    line = line.rstrip()
                    if len(line) > 0:
                        if line[0] != '#':
                            collab, data_path = line.split(',', maxsplit=1)
                            plan.cols_data_paths[collab] = data_path

            if resolve:
                plan.resolve()

                Plan.logger.info(
                    f'Parsing Federated Learning Plan : [green]SUCCESS[/] : '
                    f'[blue]{plan_config_path}[/].',
                    extra={'markup': True})
                Plan.logger.info(dump(plan.config))

            return plan

        except Exception:
            Plan.logger.exception(f'Parsing Federated Learning Plan : '
                                  f'[red]FAILURE[/] : [blue]{plan_config_path}[/].',
                                  extra={'markup': True})
            raise

    def get_aggregator(self, tensor_dict=None):
        """Get federation aggregator."""
        defaults = self.config.get('aggregator',
                                   {
                                       TEMPLATE: 'openfl.component.Aggregator',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid
        defaults[SETTINGS]['authorized_cols'] = self.authorized_cols
        defaults[SETTINGS]['assigner'] = self.get_assigner()
        defaults[SETTINGS]['compression_pipeline'] = self.get_tensor_pipe()
        log_metric_callback = defaults[SETTINGS].get('log_metric_callback')

        if log_metric_callback:
            if isinstance(log_metric_callback, dict):
                log_metric_callback = Plan.import_(**log_metric_callback)
            elif not callable(log_metric_callback):
                raise TypeError(f'log_metric_callback should be callable object '
                                f'or be import from code part, get {log_metric_callback}')

        defaults[SETTINGS]['log_metric_callback'] = log_metric_callback
        if self.aggregator_ is None:
            Plan.logger.info(f'{defaults}')
            self.aggregator_ = Plan.build(**defaults, initial_tensor_dict=tensor_dict)
        return self.aggregator_

    def get_collaborator(self, collaborator_name, root_certificate=None, private_key=None,
                         certificate=None, task_runner=None, client=None, shard_descriptor=None):
        """Get collaborator."""
        defaults = self.config.get(
            'collaborator',
            {
                TEMPLATE: 'openfl.component.Collaborator',
                SETTINGS: {}
            }
        )
        defaults[SETTINGS]['collaborator_name'] = collaborator_name
        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid

        if task_runner is not None:
            defaults[SETTINGS]['task_runner'] = task_runner
        else:
            # Here we support new interactive api as well as old task_runner subclassing interface
            # If Task Runner class is placed incide openfl `task-runner` subpackage it is
            # a part of the New API and it is a part of OpenFL kernel.
            # If Task Runner is placed elsewhere, somewhere in user workspace, than it is
            # a part of the old interface and we follow legacy initialization procedure.
            # TODO This is not general
            if 'openfl.federated.task.task_runner' in self.config['task_runner']['template'] or \
                    'unito.openfl_ext.runner_generic.GenericTaskRunner' in self.config['task_runner']['template']:
                # Interactive API
                model_provider, task_keeper, data_loader = self.deserialize_interface_objects()
                data_loader = self.initialize_data_loader(data_loader, shard_descriptor)
                defaults[SETTINGS]['task_runner'] = self.get_core_task_runner(
                    data_loader=data_loader,
                    model_provider=model_provider,
                    task_keeper=task_keeper)
            else:
                # TaskRunner subclassing API
                data_loader = self.get_data_loader(collaborator_name)
                defaults[SETTINGS]['task_runner'] = self.get_task_runner(data_loader)

        defaults[SETTINGS]['compression_pipeline'] = self.get_tensor_pipe()
        defaults[SETTINGS]['task_config'] = self.config.get('tasks', {})
        if client is not None:
            defaults[SETTINGS]['client'] = client
        else:
            defaults[SETTINGS]['client'] = self.get_client(
                collaborator_name,
                self.aggregator_uuid,
                self.federation_uuid,
                root_certificate,
                private_key,
                certificate
            )

        if self.collaborator_ is None:
            self.collaborator_ = Plan.build(**defaults)

        return self.collaborator_

    def get_core_task_runner(self, data_loader=None,
                             model_provider=None,
                             task_keeper=None):
        """Get task runner."""
        defaults = self.config.get(
            'task_runner',
            {
                TEMPLATE: 'openfl.federated.task.task_runner.CoreTaskRunner',
                SETTINGS: {}
            })
        if self.runner_ is None:
            self.runner_ = Plan.build(**defaults)

        self.runner_.set_data_loader(data_loader)

        self.runner_.set_model_provider(model_provider)
        self.runner_.set_task_provider(task_keeper)

        framework_adapter = Plan.build(
            self.config['task_runner']['required_plugin_components']['framework_adapters'], {})

        # This step initializes tensorkeys
        # Which have no sens if task provider is not set up
        self.runner_.set_framework_adapter(framework_adapter)

        return self.runner_

    def get_tensor_pipe(self):
        """Get data tensor pipeline."""
        defaults = self.config.get(
            'compression_pipeline',
            {
                TEMPLATE: 'openfl.pipelines.NoCompressionPipeline',
                SETTINGS: {}
            }
        )

        if self.pipe_ is None:
            self.pipe_ = Plan.build(**defaults)

        return self.pipe_

    def interactive_api_get_server(self, *, tensor_dict, root_certificate, certificate,
                                   private_key, tls):
        """Get gRPC server of the aggregator instance."""
        server_args = self.config['network'][SETTINGS]

        # patch certificates
        server_args['root_certificate'] = root_certificate
        server_args['certificate'] = certificate
        server_args['private_key'] = private_key
        server_args['tls'] = tls

        server_args['aggregator'] = self.get_aggregator(tensor_dict)
        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_

    def deserialize_interface_objects(self):
        """Deserialize objects for TaskRunner."""
        api_layer = self.config['api_layer']
        filenames = [
            'model_interface_file',
            'tasks_interface_file',
            'dataloader_interface_file'
        ]
        return (self.restore_object(api_layer['settings'][filename]) for filename in filenames)

    def restore_object(self, filename):
        """Deserialize an object."""
        serializer_plugin = self.get_serializer_plugin()
        if serializer_plugin is None:
            return None
        obj = serializer_plugin.restore_object(filename)
        return obj
