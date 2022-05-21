from importlib import import_module
from os.path import splitext

from openfl.component import RandomGroupedAssigner


class Assigner(RandomGroupedAssigner):

    def __init__(self, task_groups, **kwargs):
        super().__init__(task_groups, **kwargs)

    def get_aggregation_type_for_task(self, task_name):
        """Extract aggregation type from self.tasks."""
        if 'aggregation_type' not in self.tasks[task_name]:
            return None

        template = self.tasks[task_name]['aggregation_type']
        # TODO: Very bad solution, should be integrated better with aggregator
        if isinstance(template, str):
            class_name = splitext(template)[1].strip('.')
            module_path = splitext(template)[0]
            module = import_module(module_path)
            # TODO: Make it possible to pass arguments to this
            template = getattr(module, class_name)()

        return template
