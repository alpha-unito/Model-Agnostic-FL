from openfl.native import *

from .plan import Plan


def run_experiment(collaborator_dict: dict, override_config: dict = None, nn: bool = False):
    """
    Core function that executes the FL Plan.

    Args:
        collaborator_dict : dict {collaborator_name(str): FederatedModel}
            This dictionary defines which collaborators will participate in the
            experiment, as well as a reference to that collaborator's
            federated model.
        override_config : dict {flplan.key : flplan.value}
            Override any of the plan parameters at runtime using this
            dictionary. To get a list of the available options, execute
            `fx.get_plan()`
        nn : True if the model is a neural network, False otherwise

    Returns:
        final_federated_model : FederatedModel
            The final model resulting from the federated learning experiment
    """
    from sys import path

    if override_config is None:
        override_config = {}

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    # Update the plan if necessary
    plan = update_plan(override_config)
    # Overwrite plan values
    plan.authorized_cols = list(collaborator_dict)
    tensor_pipe = plan.get_tensor_pipe()

    # This must be set to the final index of the list (this is the last
    # tensorflow session to get created)
    plan.runner_ = list(collaborator_dict.values())[-1]
    model = plan.runner_

    # Initialize model weights
    init_state_path = plan.config['aggregator']['settings']['init_state_path']
    rounds_to_train = plan.config['aggregator']['settings']['rounds_to_train']
    if nn:
        tensor_dict, holdout_params = split_tensor_dict_for_holdouts(
            logger,
            model.get_tensor_dict(False)
        )

        model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
                                                 round_number=0,
                                                 tensor_pipe=tensor_pipe,
                                                 nn=nn)

        logger.info(f'Creating Initial Weights File    🠆 {init_state_path}')

        utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    logger.info('Starting Experiment...')

    aggregator = plan.get_aggregator()

    # Create the collaborators
    collaborators = {
        collaborator: create_collaborator(
            plan, collaborator, collaborator_dict[collaborator], aggregator, nn
        ) for collaborator in plan.authorized_cols
    }

    for _ in range(rounds_to_train):
        for col in plan.authorized_cols:
            collaborator = collaborators[col]
            collaborator.run_simulation()

    # Set the weights for the final model
    if nn:
        model.rebuild_model(
            rounds_to_train - 1, aggregator.last_tensor_dict, validation=True)
    return model


def create_collaborator(plan, name, model, aggregator, nn):
    """
    Create the collaborator.

    Using the same plan object to create multiple collaborators leads to
    identical collaborator objects. This function can be removed once
    collaborator generation is fixed in openfl/federated/plan/plan.py
    """
    plan = copy(plan)

    return plan.get_collaborator(name, task_runner=model, client=aggregator, nn=nn)


def update_plan(override_config):
    """
    Update the plan with the provided override and save it to disk.

    For a list of available override options, call `fx.get_plan()`

    Args:
        override_config : dict {"COMPONENT.settings.variable" : value}

    Returns:
        None
    """
    plan = setup_plan()
    flat_plan_config = flatten(plan.config, return_complete=True)
    for k, v in override_config.items():
        if k in flat_plan_config:
            logger.info(f'Updating {k} to {v}... ')
        else:
            # TODO: We probably need to validate the new key somehow
            logger.warn(f'Did not find {k} in config. Make sure it should exist. Creating...')
        flat_plan_config[k] = v
    plan.config = unflatten(flat_plan_config, '.')
    plan.resolve()
    return plan


def setup_plan(log_level='CRITICAL'):
    """
    Dump the plan with all defaults + overrides set.

    Args:
        log_level: log level to adopt

    Returns:
        plan : Plan object
    """
    plan_config = 'plan/plan.yaml'
    cols_config = 'plan/cols.yaml'
    data_config = 'plan/data.yaml'

    current_level = logging.root.level
    getLogger().setLevel(log_level)
    plan = Plan.parse(plan_config_path=Path(plan_config),
                      cols_config_path=Path(cols_config),
                      data_config_path=Path(data_config),
                      resolve=False)
    getLogger().setLevel(current_level)

    return plan
