#include "operators/elementwise.h"

namespace infini {

Binary::Binary(OperatorType type, Tensor* input_left, Tensor* input_right,
               Tensor* output)
    : Operator(type, {input_left, input_right}, {output}) {
  checkValid();
}

Binary::~Binary() {
  for (Kernel* ptr : kernel_list) {
    delete ptr;
  }
}

void Binary::applySplit() {
  inputs_tiles.clear();
  outputs_tiles.clear();
  inputs_tiles.push_back(inputs[0]->tiling(split));
  inputs_tiles.push_back(inputs[1]->tiling(split));
  outputs_tiles.push_back(outputs[0]->tiling(split));
}

bool Binary::checkValid() {
  return true;
  // TODO(wanghailu): need some check code
}

std::string Binary::generatorBone(PlatformType platform) {
  switch (platform) {
    case PlatformType::CUDA:
      return generatorBoneOnCUDA("binary");
    case PlatformType::BANG:
      return generatorBoneOnBANG("binary");
    default:
      return "UNKNOWN";
  }
}

std::string Binary::generatorBoneOnCUDA(std::string name) {
  std::string temp = "";
  // Device Function
  temp += "__device__ void " + name + "Function(";
  temp += datatype_string(inputs[0]->tensor_datatype) + "* " +
          inputs[0]->tensor_name + ", ";
  temp += datatype_string(inputs[1]->tensor_datatype) + "* " +
          inputs[1]->tensor_name + ", ";
  temp += datatype_string(outputs[0]->tensor_datatype) + "* " +
          outputs[0]->tensor_name + ")";
  temp += " {\n";
  temp += indentation(1) + "switch(blockIdx.x) {\n";
  for (auto i = 0; i < worker_list.size(); ++i) {
    temp += indentation(2) + "case " + std::to_string(i) + " :\n";
    temp += generatorCoreOnCUDA(i);
    temp += indentation(3) + "break;\n";
  }
  temp += indentation(2) + "default:\n";
  temp += indentation(3) + "break;\n";
  temp += indentation(1) + "}\n";
  temp += "}\n";
  // Global Function
  temp += "__global__ void " + name + "Kernel(";
  temp += datatype_string(inputs[0]->tensor_datatype) + "* " +
          inputs[0]->tensor_name + ", ";
  temp += datatype_string(inputs[1]->tensor_datatype) + "* " +
          inputs[1]->tensor_name + ", ";
  temp += datatype_string(outputs[0]->tensor_datatype) + "* " +
          outputs[0]->tensor_name + ")";
  temp += " {\n";
  temp += indentation(1) + name + "Function(";
  temp += inputs[0]->tensor_name + ", ";
  temp += inputs[1]->tensor_name + ", ";
  temp += outputs[0]->tensor_name + ");\n";
  temp += "}\n";
  return temp;
}

std::string Binary::generatorBoneOnBANG(std::string name) {
  std::string temp = "";
  // Device Function
  temp += "__mlu_device__ void " + name + "Function(";
  temp += datatype_string(inputs[0]->tensor_datatype) + "* " +
          inputs[0]->tensor_name + ", ";
  temp += datatype_string(inputs[1]->tensor_datatype) + "* " +
          inputs[1]->tensor_name + ", ";
  temp += datatype_string(outputs[0]->tensor_datatype) + "* " +
          outputs[0]->tensor_name + ")";
  temp += " {\n";
  temp += indentation(1) + "switch(taskId) {\n";
  for (auto i = 0; i < worker_list.size(); ++i) {
    temp += indentation(2) + "case " + std::to_string(i) + " :\n";
    temp += generatorCoreOnBANG(i);
    temp += indentation(3) + "break;\n";
  }
  temp += indentation(2) + "default:\n";
  temp += indentation(3) + "break;\n";
  temp += indentation(1) + "}\n";
  temp += "}\n";
  // Global Function
  temp += "__mlu_global__ void " + name + "Kernel(";
  temp += datatype_string(inputs[0]->tensor_datatype) + "* " +
          inputs[0]->tensor_name + ", ";
  temp += datatype_string(inputs[1]->tensor_datatype) + "* " +
          inputs[1]->tensor_name + ", ";
  temp += datatype_string(outputs[0]->tensor_datatype) + "* " +
          outputs[0]->tensor_name + ")";
  temp += " {\n";
  temp += indentation(1) + name + "Function(";
  temp += inputs[0]->tensor_name + ", ";
  temp += inputs[1]->tensor_name + ", ";
  temp += outputs[0]->tensor_name + ");\n";
  temp += "}\n";
  return temp;
}

std::string ADD::generatorCoreOnCUDA(int64_t id) {
  std::string temp = "";
  temp += indentation(3) + "cuda" + ";\n";
  return temp;
  // TODO(wanghailu)
}

std::string ADD::generatorCoreOnBANG(int64_t id) {
  std::string temp = "";
  temp += indentation(3) + datatype_string(inputs[0]->tensor_datatype) + "* " +
          inputs[0]->tensor_name + "_start = " + inputs[0]->tensor_name +
          " + " + std::to_string(inputs_tiles[0][id].start_offset) + ";\n";
  temp += indentation(3) + datatype_string(inputs[1]->tensor_datatype) + "* " +
          inputs[1]->tensor_name + "_start = " + inputs[1]->tensor_name +
          " + " + std::to_string(inputs_tiles[1][id].start_offset) + ";\n";
  temp += indentation(3) + datatype_string(outputs[0]->tensor_datatype) + "* " +
          outputs[0]->tensor_name + "_start = " + outputs[0]->tensor_name +
          " + " + std::to_string(outputs_tiles[0][id].start_offset) + ";\n";
  temp += indentation(3) + "int64_t repeat = " +
          std::to_string(VECTOR_PRODUCT(inputs_tiles[0][id].tile_dimension)) +
          " / " + std::to_string(worker_list[id]->cache_line_size) + ";\n";
  temp += indentation(3) + "int64_t rem = " +
          std::to_string(VECTOR_PRODUCT(inputs_tiles[0][id].tile_dimension)) +
          " % " + std::to_string(worker_list[id]->cache_line_size) + ";\n";
  temp += indentation(3) + "for (int64_t i = 0; i < repeat; ++i) {\n";
  for (auto i = 0; i < kernel_list.size(); ++i) {
    temp +=
        indentation(4) +
        kernel_list[i]->generatorCodeOnBANG(
            {"placeholder0", "placeholder1", "placeholder2", "placeholder3"}) +
        "\n";
  }
  temp += indentation(4) + inputs[0]->tensor_name +
          "_start += " + std::to_string(worker_list[id]->cache_line_size) +
          ";\n";
  temp += indentation(4) + inputs[1]->tensor_name +
          "_start += " + std::to_string(worker_list[id]->cache_line_size) +
          ";\n";
  temp += indentation(4) + outputs[0]->tensor_name +
          "_start += " + std::to_string(worker_list[id]->cache_line_size) +
          ";\n";
  temp += indentation(3) + "}\n";
  temp += indentation(3) + "if (rem) {\n";
  for (auto i = 0; i < kernel_list.size(); ++i) {
    temp +=
        indentation(4) +
        kernel_list[i]->generatorCodeOnBANG(
            {"placeholder0", "placeholder1", "placeholder2", "placeholder3"}) +
        "\n";
  }
  temp += indentation(3) + "}\n";
  return temp;
}

}  // namespace infini
