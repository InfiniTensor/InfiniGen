class Operator:
    num_instances = 0

    def __init__(self, inputs, outputs, operation):
        self.inputs = inputs
        self.outputs = outputs
        self.operation = operation
        self.predecessors = []
        self.successors = []
        self.name = f"operator_{type(self).num_instances}"

        for input in self.inputs:
            input.consumers.append(self)

            if input.producer is not None:
                self.predecessors.append(input.producer)
                input.producer.successors.append(self)

        for output in self.outputs:
            output.producer = self

        type(self).num_instances += 1

    def __repr__(self):
        return f"{self.inputs} -> {self.name} -> {self.outputs}"
