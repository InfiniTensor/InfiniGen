#include "core/tensor.h"
#include "core/utils.h"
#include "core/type.h"

namespace infini {

Tensor::Tensor(const std::vector<int64_t>& dimension, TensorDatatype dtype,
               TensorType type, TensorLayout layout, std::string name,
               int64_t offset) {
  tensor_dimension = dimension;
  tensor_datatype = dtype;
  tensor_type = type;
  tensor_layout = layout;
  tensor_name = name;
  data_offset = offset;
  is_contiguous = true;
  tensor_stride = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = tensor_stride.size() - 2; i >= 0; --i) {
    tensor_stride[i] = tensor_stride[i + 1] * tensor_dimension[i + 1];
  }
}

Tensor::Tensor(const std::vector<int64_t>& dimension,
               const std::vector<int64_t>& stride, TensorDatatype dtype,
               TensorType type, TensorLayout layout, std::string name,
               int64_t offset) {
  tensor_dimension = dimension;
  tensor_stride = stride;
  tensor_datatype = dtype;
  tensor_type = type;
  tensor_layout = layout;
  tensor_name = name;
  data_offset = offset;
  std::vector<int64_t> temp = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = temp.size() - 2; i >= 0; --i) {
    temp[i] = temp[i + 1] * tensor_dimension[i + 1];
  }
  is_contiguous = ALL(temp == tensor_dimension);
}

TileTensor Tensor::tiling(const Split& split) {
  // Check
  CHECK_EQ(tensor_dimension.size(), split.split_dimension.size());
  std::vector<int64_t> easy = tensor_dimension / split.split_dimension;
  std::vector<int64_t> boundary = tensor_dimension % split.split_dimension;
  std::vector<int64_t> heavy(tensor_dimension.size(), 0);
  for (auto i = 0; i < heavy.size(); ++i) {
    heavy[i] = (boundary[i] == 0 ? easy[i] : easy[i] + 1);
  }
  std::vector<int64_t> split_suffix(split.split_dimension.size(), 1);
  for (int64_t i = split.split_dimension.size() - 2; i >= 0; --i) {
    split_suffix[i] = split_suffix[i + 1] * split.split_dimension[i + 1];
  }
  int64_t total = VECTOR_PRODUCT(split.split_dimension);
  TileTensor result(split.split_dimension, split_suffix, tensor_type,
                    tensor_layout, tensor_name + "_split");
  for (int64_t i = 0; i < total; ++i) {
    // Local Position
    int64_t pos = i;
    int64_t axis = 0;
    std::vector<int64_t> tile_local_position;
    while (axis < split_suffix.size()) {
      tile_local_position.push_back(pos / split_suffix[axis]);
      pos %= split_suffix[axis];
      ++axis;
    }
    // Dimension
    std::vector<int64_t> tile_dimension(split.split_dimension.size(), 0);
    for (auto j = 0; j < tile_dimension.size(); ++j) {
      tile_dimension[j] =
          (tile_local_position[j] < boundary[j] ? heavy[j] : easy[j]);
    }
    // Stride
    std::vector<int64_t> tile_stride = tensor_stride;
    // Start Position
    std::vector<int64_t> start_position(tile_dimension.size(), 0);
    for (auto j = 0; j < tile_dimension.size(); ++j) {
      start_position[j] =
          (tile_local_position[j] <= boundary[j])
              ? (heavy[j] * tile_local_position[j])
              : (heavy[j] * boundary[j] +
                 (tile_local_position[j] - boundary[j]) * easy[j]);
    }
    // Offset
    int64_t tile_start = 0;
    for (auto j = 0; j < tensor_dimension.size(); ++j) {
      tile_start += start_position[j] * tensor_stride[j];
    }
    std::string tile_name = TO_STRING(tile_local_position) + " tile of " +
                            tensor_name + " with global start position " +
                            TO_STRING(start_position);
    Tile temp(tile_dimension, tile_local_position, tile_stride, tile_name,
              tile_start);
    result.addTile(temp);
  }

  return result;
}

// TileTensor Tensor::tiling(const Tile & t){
//   CHECK_EQ(tensor_dimension.size(), t.tile_dimension.size());
//   // This stride means every k step in this dimmension
//   std::vector<int64_t> tile_stride = t.tile_stride;
//   std::vector<int64_t> split_dim = tensor_dimension / t.tile_dimension;
//   // Tile stride restrict
//   ASSERT(ALL_EQLESS(tile_stride, split_dim));

// }

void Tensor::printInformation() {
  std::string info_string = "";
  info_string += "—— Tensor ";
  info_string += "Name: ";
  info_string += tensor_name;
  info_string += " ";
  info_string += "Datatype: ";
  info_string += TO_STRING(tensor_datatype);
  info_string += " ";
  info_string += "Type: ";
  info_string += TO_STRING(tensor_type);
  info_string += " ";
  info_string += "Layout: ";
  info_string += TO_STRING(tensor_layout);
  info_string += " ";
  info_string += "Dimension: ";
  info_string += TO_STRING(tensor_dimension);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(tensor_stride);
  info_string += " ";
  info_string += "Offset: ";
  info_string += std::to_string(data_offset);
  LOG(INFO) << info_string;
}

void Tensor::printSummary() {
  std::string info_string = "";
  info_string += "Tensor ";
  info_string += "Dtype: ";
  info_string += TO_STRING(tensor_datatype);
  info_string += " ";
  info_string += "Layout: ";
  info_string += TO_STRING(tensor_layout);
  info_string += " ";
  info_string += "Dim: ";
  info_string += TO_STRING(tensor_dimension);
  info_string += " ";
  info_string += "Stride: ";
  info_string += TO_STRING(tensor_stride);
  info_string += " ";
  info_string += "Offset: ";
  info_string += std::to_string(data_offset);
  info_string += "\n";
  LOG(PURE) << info_string;
}

bool Tensor::isContiguous() { return is_contiguous; }

void Tensor::flatten(int64_t start, int64_t end) {
  // Check
  int64_t len = tensor_dimension.size();
  CHECK(isContiguous());
  CHECK_GE(start, -len);
  CHECK_LE(start, len - 1);
  CHECK_GE(end, -len);
  CHECK_LE(end, len - 1);
  // Compute
  start = (start + len) % len;
  end = (end + len) % len;
  CHECK_LE(start, end);
  if (start == end) {
    return;
  }
  std::vector<int64_t> result_dimension(len - (end - start), 0);
  for (auto i = 0; i < start; ++i) {
    result_dimension[i] = tensor_dimension[i];
  }
  int64_t accumulate = 1;
  for (auto i = start; i <= end; ++i) {
    accumulate *= tensor_dimension[i];
  }
  result_dimension[start] = accumulate;
  for (auto i = end + 1; i < len; ++i) {
    result_dimension[++start] = tensor_dimension[i];
  }
  // Assign
  tensor_dimension = result_dimension;
  tensor_stride = std::vector<int64_t>(tensor_dimension.size(), 1);
  for (int64_t i = tensor_stride.size() - 2; i >= 0; --i) {
    tensor_stride[i] = tensor_stride[i + 1] * tensor_dimension[i + 1];
  }
}

}  // namespace infini
