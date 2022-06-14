import os
from copy import deepcopy
from pathlib import Path

from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.interface.interactive_api.experiment import TaskInterface
from openfl.utilities import split_tensor_dict_for_holdouts

from unito.openfl_ext.plan import Plan


class ModelStatus:
    """Model statuses."""

    INITIAL = 'initial'
    BEST = 'best'
    LAST = 'last'
    RESTORED = 'restored'


class FLExperiment(FLExperiment):

    def __init__(
            self,
            federation,
            experiment_name: str = None,
            serializer_plugin: str = 'openfl.plugins.interface_serializer.'
                                     'cloudpickle_serializer.CloudpickleSerializer'
    ) -> None:
        super().__init__(federation=federation, experiment_name=experiment_name, serializer_plugin=serializer_plugin)

    def start(self, *, model_provider, task_keeper, data_loader,
              rounds_to_train, delta_updates=False, opt_treatment='RESET',
              device_assignment_policy='CPU_ONLY', nn=True, default=False):
        """Prepare experiment and run."""
        self._prepare_plan(model_provider, task_keeper, data_loader,
                           rounds_to_train,
                           delta_updates=delta_updates, opt_treatment=opt_treatment,
                           device_assignment_policy=device_assignment_policy,
                           model_interface_file='model_obj.pkl',
                           tasks_interface_file='tasks_obj.pkl',
                           dataloader_interface_file='loader_obj.pkl',
                           default=default)

        self.prepare_workspace_distribution(
            model_provider, task_keeper, data_loader
        )

        self.logger.info('Starting experiment!')
        self.plan.resolve()
        initial_tensor_dict = self._get_initial_tensor_dict(model_provider)
        try:
            response = self.federation.dir_client.set_new_experiment(
                name=self.experiment_name,
                col_names=self.plan.authorized_cols,
                arch_path=self.arch_path,
                initial_tensor_dict=initial_tensor_dict
            )
        finally:
            self.remove_workspace_archive()

        if response.accepted:
            self.logger.info('Experiment was accepted and launched.')
            self.experiment_accepted = True
        else:
            self.logger.info('Experiment was not accepted or failed.')

    def _get_initial_tensor_dict(self, model_provider, nn=False):  # TODO pass the nn parameter from outside
        """Extract initial weights from the model."""
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        self.current_model_status = ModelStatus.INITIAL

        if nn:
            tensor_dict, _ = split_tensor_dict_for_holdouts(
                self.logger,
                self.task_runner_stub.get_tensor_dict(False),
                **self.task_runner_stub.tensor_dict_split_fn_kwargs
            )
        else:
            tensor_dict = self.task_runner_stub.get_tensor_dict()

        return tensor_dict

    def _prepare_plan(self, model_provider, task_keeper, data_loader,
                      rounds_to_train,
                      delta_updates, opt_treatment,
                      device_assignment_policy,
                      model_interface_file='model_obj.pkl', tasks_interface_file='tasks_obj.pkl',
                      dataloader_interface_file='loader_obj.pkl',
                      aggregation_function_interface_file='aggregation_function_obj.pkl',
                      default=False):
        """Fill plan.yaml file using provided setting."""
        # Create a folder to store plans
        os.makedirs('./plan', exist_ok=True)
        os.makedirs('./save', exist_ok=True)
        # Load the default plan
        # base_plan_path = WORKSPACE / 'workspace/plan/plans/default/base_plan_interactive_api.yaml'
        # plan = Plan.parse(base_plan_path, resolve=False)
        plan = Plan.parse(Path("./plan/plan.yaml"),
                          resolve=False)  # TODO Why do you want to overwrite the plan every time with its default?
        # Change plan name to default one
        plan.name = 'plan.yaml'

        # Seems like we still need to fill authorized_cols list
        # So aggregator know when to start sending tasks
        # We also could change the aggregator logic so it will send tasks to aggregator
        # as soon as it connects. This change should be a part of a bigger PR
        # brining in fault tolerance changes
        shard_registry = self.federation.get_shard_registry()
        plan.authorized_cols = [
            name for name, info in shard_registry.items() if info['is_online']
        ]
        # Network part of the plan
        # We keep in mind that an aggregator FQND will be the same as the directors FQDN
        # We just choose a port randomly from plan hash
        director_fqdn = self.federation.director_node_fqdn.split(':')[0]  # We drop the port
        plan.config['network']['settings']['agg_addr'] = director_fqdn
        plan.config['network']['settings']['tls'] = self.federation.tls

        # Aggregator part of the plan
        plan.config['aggregator']['settings']['rounds_to_train'] = rounds_to_train

        # Collaborator part
        plan.config['collaborator']['settings']['delta_updates'] = delta_updates
        plan.config['collaborator']['settings']['opt_treatment'] = opt_treatment
        plan.config['collaborator']['settings'][
            'device_assignment_policy'] = device_assignment_policy

        # DataLoader part
        # for setting, value in data_loader.kwargs.items():
        #    plan.config['data_loader']['settings'][setting] = value

        # Tasks part
        # @TODO This is the responsable for the overriding of the provided plan - should be integrated in a smarter way
        # @TODO this piece of code gives problems, puts the entries in the config file in alphabetical order
        # for name in task_keeper.task_registry:
        #    if task_keeper.task_contract[name]['optimizer'] is not None:
        #        # TODO Why training is defined by the presence of the optimizer?
        #        # This is training task
        #        plan.config['tasks'][name] = {'function': name,
        #                                      'kwargs': task_keeper.task_settings[name]}
        #
        #    else:
        #        # This is a validation type task (not altering the model state)
        #        for name_prefix, apply_kwarg in zip(['localy_tuned_model_', 'aggregated_model_'],
        #                                            ['local', 'global']):
        #            # We add two entries for this task: for local and global models
        #            task_kwargs = deepcopy(task_keeper.task_settings[name])
        #            task_kwargs.update({'apply': apply_kwarg})
        #            plan.config['tasks'][name_prefix + name] = {
        #                'function': name,
        #                'kwargs': task_kwargs}

        # TaskRunner framework plugin
        # ['required_plugin_components'] should be already in the default plan with all the fields
        # filled with the default values
        plan.config['task_runner']['required_plugin_components'] = {
            'framework_adapters': model_provider.framework_plugin
        }

        # API layer
        plan.config['api_layer'] = {
            'required_plugin_components': {
                'serializer_plugin': self.serializer_plugin
            },
            'settings': {
                'model_interface_file': model_interface_file,
                'tasks_interface_file': tasks_interface_file,
                'dataloader_interface_file': dataloader_interface_file,
                'aggregation_function_interface_file': aggregation_function_interface_file
            }
        }

        plan.config['assigner']['settings']['task_groups'][0]['tasks'] = [
            entry
            for entry in plan.config['tasks']
            if (type(plan.config['tasks'][entry]) is dict
                and 'function' in plan.config['tasks'][entry])
        ]
        self.plan = deepcopy(plan)

    def stream_metrics(self, tensorboard_logs: bool = True) -> None:
        """Stream metrics."""
        self._assert_experiment_accepted()
        for metric_message_dict in self.federation.dir_client.stream_metrics(self.experiment_name):
            print(metric_message_dict)
            self.logger.metric(
                f'Round {metric_message_dict["round"]}, '
                f'collaborator {metric_message_dict["metric_origin"]} '
                f'{metric_message_dict["task_name"]} result '
                f'{metric_message_dict["metric_name"]}:\t{metric_message_dict["metric_value"]}')

            if tensorboard_logs:
                self.write_tensorboard_metric(metric_message_dict)


class TaskInterface(TaskInterface):
    """
    Task keeper class.

    Task should accept the following entities that exist on collaborator nodes:
    1. model - will be rebuilt with relevant weights for every task by `TaskRunner`
    2. data_loader - data loader equipped with `repository adapter` that provides local data
    3. device - a device to be used on collaborator machines
    4. optimizer (optional)

    Task returns a dictionary {metric name: metric value for this task}
    """

    # @TODO: too much ad-hoc
    def register_fl_task(self, model, data_loader, device, optimizer=None, adaboost_coeff=None):
        """
        Register FL tasks.

        The task contract should be set up by providing variable names:
        [model, data_loader, device] - necessarily
        and optimizer - optionally

        All tasks should accept contract entities to be run on collaborator node.
        Moreover we ask users return dict{'metric':value} in every task
        `
        TI = TaskInterface()

        task_settings = {
            'batch_size': 32,
            'some_arg': 228,
        }
        @TI.add_kwargs(**task_settings)
        @TI.register_fl_task(model='my_model', data_loader='train_loader',
                device='device', optimizer='my_Adam_opt')
        def foo_task(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
            ...
        `
        """

        # The highest level wrapper for allowing arguments for the decorator
        def decorator_with_args(training_method):
            # We could pass hooks to the decorator
            # @functools.wraps(training_method)

            def wrapper_decorator(**task_keywords):
                metric_dict = training_method(**task_keywords)
                return metric_dict

            # Saving the task and the contract for later serialization
            self.task_registry[training_method.__name__] = wrapper_decorator
            contract = {'model': model, 'data_loader': data_loader,
                        'device': device, 'optimizer': optimizer,
                        'adaboost_coeff': adaboost_coeff}
            self.task_contract[training_method.__name__] = contract
            # We do not alter user environment
            return training_method

        return decorator_with_args
