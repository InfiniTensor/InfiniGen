from graphlib import TopologicalSorter


class Graph:
    num_instances = 0

    def __init__(self, operators, inputs, outputs):
        self.operators = operators
        self.inputs = inputs
        self.outputs = outputs
        self.name = f"graph_{type(self).num_instances}"

        type(self).num_instances += 1

    def topological_sort(self):
        topological_sorter = TopologicalSorter()

        for operator in self.operators:
            topological_sorter.add(operator, *operator.predecessors)

        return list(topological_sorter.static_order())
