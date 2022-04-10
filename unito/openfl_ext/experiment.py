import os
from copy import deepcopy

from openfl.interface.cli_helper import WORKSPACE
from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.utilities import split_tensor_dict_for_holdouts

from unito.openfl_ext.plan import Plan


class ModelStatus:
    """Model statuses."""

    INITIAL = 'initial'
    BEST = 'best'
    LAST = 'last'
    RESTORED = 'restored'


class FLExperiment(FLExperiment):

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
        base_plan_path = WORKSPACE / 'workspace/plan/plans/default/base_plan_interactive_api.yaml'
        plan = Plan.parse(base_plan_path, resolve=False)
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
        for setting, value in data_loader.kwargs.items():
            plan.config['data_loader']['settings'][setting] = value

        # Tasks part
        for name in task_keeper.task_registry:
            if task_keeper.task_contract[name]['optimizer'] is not None:
                # TODO Why training is defined by the presence of the optimizer?
                # This is training task
                plan.config['tasks'][name] = {'function': name,
                                              'kwargs': task_keeper.task_settings[name]}
            else:
                # This is a validation type task (not altering the model state)
                for name_prefix, apply_kwarg in zip(['localy_tuned_model_', 'aggregated_model_'],
                                                    ['local', 'global']):
                    # We add two entries for this task: for local and global models
                    task_kwargs = deepcopy(task_keeper.task_settings[name])
                    task_kwargs.update({'apply': apply_kwarg})
                    plan.config['tasks'][name_prefix + name] = {
                        'function': name,
                        'kwargs': task_kwargs}

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
