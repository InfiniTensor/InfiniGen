from infinigen.arch.function import Type as FunctionType
from infinigen.tensor import DataType


class CUDA:
    def func_decl(self, name, params, func_type):
        param_decls = ", ".join(
            f"{self._dtype(param.dtype)}* {param.name}" for param in params
        )

        if func_type == FunctionType.DEVICE:
            specifier = "__device__"
        else:
            specifier = ""
        if len(specifier) != 0:
            specifier += " "

        return f"{specifier}void {name}({param_decls})"

    def add(self, lhs, rhs, output):
        return self._binary(lhs, rhs, output, "+")

    def sub(self, lhs, rhs, output):
        return self._binary(lhs, rhs, output, "-")

    def _binary(self, lhs, rhs, output, operator):
        return f"{self._access(output)} = {self._access(lhs)} {operator} {self._access(rhs)}"

    def _access(self, tensor):
        return f"{tensor.original.name}[{sum(tensor.offsets())}]"

    def _dtype(self, dtype):
        if dtype == DataType.FLOAT32:
            return "float"

        raise ValueError
