#include "core/platform.h"
#include "core/log.h"
#include "core/utils.h"

namespace infini {

#define CASE(TYPE, STR)                                                        \
    case Platform::TYPE:                                                       \
        return STR

const std::string Platform::deviceFuncDecl(std::string name) const {
    switch (type) {
        CASE(CUDA, "__device__ void " + name);
        CASE(BANG, "__mlu_func__ void " + name);
        CASE(ASCEND, "__aicore__ inline void " + name);
        CASE(KUNLUN, "__device__ void " + name);
    default:
        return "";
    }
}

const std::string Platform::globalFuncDecl(std::string name) const {
    switch (type) {
        CASE(CUDA, "__global__ void " + name);
        CASE(BANG, "__mlu_entry__ void " + name);
        CASE(ASCEND, "extern \"C\" __global__ __aicore__ void " + name);
        CASE(KUNLUN, "__global__ void " + name);
    default:
        return "";
    }
}

const std::string Platform::threadId(int dim) const {
    // Coordinates of thread within a block, cuda only
    std::vector<std::string> dim_map_cuda = {".x", ".y", ".z"};
    switch (type) {
        CASE(CUDA, "threadIdx" + dim_map_cuda[dim]);
    default:
        return "";
    }
}

const std::string Platform::threadId() const {
    // Linear ID of thread within a block, cuda only
    switch (type) {
        CASE(CUDA, "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * "
                   "blockDim.x * blockDim.y)");
    default:
        return "";
    }
}

const std::string Platform::taskId(int dim) const {
    // Coordinates of block/task on device
    std::vector<std::string> dim_map_cuda = {".x", ".y", ".z"};
    std::vector<std::string> dim_map_bang = {"X", "Y", "Z"};
    switch (type) {
        CASE(CUDA, "blockIdx" + dim_map_cuda[dim]);
        CASE(BANG, "taskId" + dim_map_bang[dim]);
    default:
        return "";
    }
}

const std::string Platform::taskId() const {
    // Linear ID of block/task on device
    switch (type) {
        CASE(CUDA,
             "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * "
             "gridDim.y)");
        CASE(BANG, "taskId");
        CASE(ASCEND, "GetBlockIdx()");
        CASE(KUNLUN, "(core_num() * cluster_id() + core_id())");
    default:
        return "";
    }
}

const std::string Platform::taskDim(int dim) const {
    // Number of blocks/tasks on each dimension
    std::vector<std::string> dim_map_cuda = {".x", ".y", ".z"};
    std::vector<std::string> dim_map_bang = {"X", "Y", "Z"};
    switch (type) {
        CASE(CUDA, "gridDim" + dim_map_cuda[dim]);
        CASE(BANG, "taskDim" + dim_map_bang[dim]);
    default:
        return "";
    }
}

const std::string Platform::taskDim() const {
    // Total number of blocks/tasks
    switch (type) {
        CASE(CUDA, "(gridDim.x * gridDim.y * gridDim.z)");
        CASE(BANG, "taskDim");
    default:
        return "";
    }
}

const std::string Platform::regDecl(std::string datatype,
                                    std::string name) const {
    switch (type) {
        CASE(CUDA, datatype + " " + name);
        CASE(BANG, "__nram__ " + datatype + " " + name);
        CASE(ASCEND, "LocalTensor<" + datatype + "> " + name);
        CASE(KUNLUN, "__local__ " + datatype + " " + name);
    default:
        return "";
    }
}

const std::string Platform::ldramDecl(std::string datatype,
                                      std::string name) const {
    switch (type) {
        CASE(CUDA, datatype + " " + name); // 不确定是不是这个
        CASE(BANG, "__ldram__ " + datatype + " " + name);
        CASE(ASCEND, "");
        CASE(KUNLUN, "");
    default:

        return "";
    }
}

const std::string Platform::shmemDecl(std::string datatype,
                                      std::string name) const {
    switch (type) {
        CASE(CUDA, "__shared__ " + datatype + " " + name);
        CASE(BANG, "__mlu_shared__ " + datatype + " " + name);
        CASE(ASCEND, "");
        CASE(KUNLUN, "__shared__ " + datatype + " " + name);
    default:
        return "";
    }
}

const std::string Platform::glmemDecl(std::string datatype,
                                      std::string name) const {
    switch (type) {
        CASE(CUDA, "__device__ " + datatype + " " + name);
        CASE(BANG, "__mlu_device__ " + datatype + " " + name);
        CASE(ASCEND, "GlobalTensor<" + datatype + "> " + name);
        CASE(KUNLUN, "__global_ptr__ " + datatype + " " + name);
    default:
        return "";
    }
}

const std::string Platform::queue() const {
    switch (type) {
        CASE(CUDA, "cudaStream_t");
        CASE(BANG, "cnrtQueue_t");
        CASE(ASCEND, "void*");
        CASE(KUNLUN, "XPUStream");
    default:
        return "";
    }
}

const std::string Platform::head() const {
    switch (type) {
        CASE(CUDA, "#include <cuda.h>");
        CASE(BANG, "#include <bang.h>");
        CASE(ASCEND,
             "#include \"kernel_operator.h\"\nusing namespace AscendC;");
        CASE(KUNLUN, "#include \"xpu/runtime.h\"\n#include "
                     "\"xpu/kernel/cluster_header.h\"\n#include "
                     "\"xpu/kernel/debug.h\"\n#include \"xpu/kernel/math.h\"");
    default:
        return "";
    }
}

const char *Platform::toString() const {
    switch (type) {
        CASE(CUDA, "CUDA");
        CASE(BANG, "BANG");
        CASE(ASCEND, "ASCEND");
        CASE(KUNLUN, "KUNLUN");
    default:
        return "Unknown";
    }
}

const std::string Platform::taskScaleDecl(Tiles tiles) const {
    // TODO: how to determine task scale
    int64_t num_cores = tiles.size();
    switch (type) {
        CASE(CUDA, "int numBlocks = " + std::to_string(num_cores) +
                       ", threadsPerBlock = " +
                       std::to_string(tiles[0]->getElementNum()) + ";");

        CASE(BANG, "cnrtDim3_t dim = {" + std::to_string(PAD_UP(num_cores, 4)) +
                       ", 1, 1};");
        CASE(ASCEND, "int numBlocks = " + std::to_string(num_cores) + ";");
        CASE(KUNLUN, "int numBlocks = " + std::to_string(num_cores) +
                         ", threadsPerBlock = 1;");
    default:
        return "";
    }
}

const std::string
Platform::taskScaleDecl(std::vector<int64_t> tileGridShape,
                        std::vector<int64_t> tileShape) const {
    // Assume taskScale is the same as tileGrid
    std::vector<int64_t> tileGridShapePadded;
    for (auto i = 0; i < 3; i++) {
        tileGridShapePadded.push_back(
            i < tileGridShape.size() ? PAD_UP(tileGridShape[i], 4) : 1);
    }

    switch (type) {
        CASE(CUDA, "dim3 numBlocks(" + TO_STRING(tileGridShape) +
                       "), threadsPerBlock(" + TO_STRING(tileShape) + ");");

        CASE(BANG,
             "cnrtDim3_t dim = " + INITIALIZER(tileGridShapePadded) + ";");
    default:
        return "";
    }
}

const std::string Platform::syntacticSugar() const {
    switch (type) {
        CASE(CUDA, "<<<numBlocks, threadsPerBlock, 0, queue>>>");
        CASE(BANG, "<<<dim, CNRT_FUNC_TYPE_UNION1, queue>>>");
        CASE(ASCEND, "<<<numBlocks, nullptr, queue>>>");
        CASE(KUNLUN, "<<<numBlocks, threadsPerBlock, queue>>>");
    default:
        return "";
    }
}

// const std::string Platform::workingCoreCond(TileTensor tiles) const {
//     switch (type) {
//         CASE(CUDA, "");
//         CASE(BANG, "if (taskId >= " + std::to_string(tiles.numNeatTiles()) +
//                        ") { return; }");
//     default:
//         return "";
//     }
// }

// const std::string Platform::remainingTileCond(TileTensor tiles) const {
//     switch (type) {
//         CASE(CUDA, taskId() + " < " + std::to_string(tiles.numRemainTiles())
//         +
//                        " && " + "threadIdx.x < " +
//                        std::to_string(VECTOR_PRODUCT(
//                            tiles.remain_tiles[0].tile_dimension)));
//         // TODO: n-d situations
//         CASE(BANG, taskId() + " < " +
//         std::to_string(tiles.numRemainTiles()));
//     default:
//         return "";
//     }
// }

const std::string Platform::offset(std::vector<int64_t> tensorStride,
                                   std::vector<int64_t> tileGridStride,
                                   std::vector<int64_t> tileShape,
                                   bool threadOffset) const {
    std::string result = "";
    std::string index = threadOffset ? threadId() : taskId();
    if (tileShape.size() == 3) {
        /* (tileId / tileGridStride[0]) * tileShape[0] * tensorStride[0] +
           (tileId % tileGridStride[0]) / tileGridStride[1] * tileShape[1] *
           tensorStride[1] + ((tileId % tileGridStride[0]) % tileGridStride[1])
           * tileShape[2] * tensorStride[2] */
        result += "((int)(" + index + " / " +
                  std::to_string(tileGridStride[0]) + ") * " +
                  std::to_string(tileShape[0]) + " * " +
                  std::to_string(tensorStride[0]) + " + " + "(int)((" + index +
                  " % " + std::to_string(tileGridStride[0]) + ") / " +
                  std::to_string(tileGridStride[1]) + ") * " +
                  std::to_string(tileShape[1]) + " * " +
                  std::to_string(tensorStride[1]) + " + ((" + index + " % " +
                  std::to_string(tileGridStride[0]) + ") % " +
                  std::to_string(tileGridStride[1]) + ") * " +
                  std::to_string(tileShape[2]) + ")";

    } else if (tileShape.size() == 2) {
        /* (tileId / tileGridStride[0]) * tileShape[0] * tensorStride[0] +
           (tileId % tileGridStride[0]) * tileShape[1] * tensorStride[1]; */
        result += "((int)(" + index + " / " +
                  std::to_string(tileGridStride[0]) + ") * " +
                  std::to_string(tileShape[0]) + " * " +
                  std::to_string(tensorStride[0]) + " + " + "(" + index +
                  " % " + std::to_string(tileGridStride[0]) + ") * " +
                  std::to_string(tileShape[1]) + ")";

    } else if (tileShape.size() == 1) {
        /* tileId * tileShape[0] */
        result += "(" + index + " * " + std::to_string(tileShape[0]) + ")";

    } else {
        LOG(ERROR) << "tileShape > 3 not supported.";
    }
    return result;
}

const std::string Platform::cacheDecl(std::string name, int64_t cache_size,
                                      std::string datatype) const {
    switch (type) {
        CASE(CUDA, "char " + name + "[" + std::to_string(cache_size) + "];");
        CASE(BANG,
             "__nram__ char " + name + "[" + std::to_string(cache_size) + "];");
        CASE(ASCEND, "TPipe pipe; TBuf<TPosition::VECCALC> tbuf;"
                     " pipe.InitBuffer(tbuf, " +
                         std::to_string(cache_size) + "); LocalTensor<" +
                         datatype + "> " + name + " = tbuf.Get<" + datatype +
                         ">();");
        CASE(KUNLUN, "__local__ char " + name + "[" +
                         std::to_string(cache_size) + "];");
    default:
        return "";
    }
}

const std::string Platform::ldramDecl(std::string name,
                                      int64_t ldram_size) const {
    switch (type) {
        CASE(CUDA, "");
        CASE(BANG, "__ldram__ char " + name + "_ldram[" +
                       std::to_string(ldram_size) + "];");
        CASE(ASCEND, "");
        CASE(KUNLUN, "");
    default:
        return "";
    }
}

bool Platform::isCUDA() const { return type == Platform::CUDA; }

bool Platform::isBANG() const { return type == Platform::BANG; }

bool Platform::isASCEND() const { return type == Platform::ASCEND; }

bool Platform::isKUNLUN() const { return type == Platform::KUNLUN; }

} // namespace infini
