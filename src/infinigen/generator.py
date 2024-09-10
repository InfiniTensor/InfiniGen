from infinigen.arch.cuda import CUDA
from infinigen.arch.function import Type as FunctionType

architectures = {"cuda": CUDA}


class CodeGenerator:
    def __init__(self, graph, tile_shape, architecture="cuda"):
        self._graph = graph
        self._tile_shape = tile_shape
        self._arch = architectures[architecture]()

        self._params = graph.inputs + graph.outputs

        self._body = []
        for operator in self._graph.topological_sort():
            operator.inputs = tuple(
                tensor.tile(self._tile_shape) for tensor in operator.inputs
            )
            operator.outputs = tuple(
                tensor.tile(self._tile_shape) for tensor in operator.outputs
            )

            self._body.append(
                getattr(self._arch, operator.operation.value)(
                    *operator.inputs, *operator.outputs
                )
            )

        self._indentation_level = 0

    def generate_source_file(self):
        self._indentation_level += 1
        statements = "\n".join(
            f"{self._indent()}{statement};" for statement in self._body
        )
        self._indentation_level -= 1

        return (
            self._generate_dependencies()
            + "\n"
            + f"{self._arch.func_decl(self._graph.name, self._params, FunctionType.DEVICE)} {{\n{statements}\n}}".replace(
                "ninetoothed.language.", "infinigen_"
            ).replace("//", "/")
        )

    def _generate_dependencies(self):
        return """
__device__ size_t infinigen_program_id(...) {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ int infinigen_cdiv(double lhs, double rhs) {
    // TODO: Make this real `cdiv`.
    return lhs / rhs;
}

__device__ size_t infinigen_arange(...) {
    return 0;
}
        """

    def _indent(self, indentation_width=4):
        return " " * indentation_width * self._indentation_level
