#include "operators/broadcast.h"
#include "core/utils.h"

namespace infini {

bool isValidExpandShape(const std::vector<int64_t> &inputShape,
                        const std::vector<int64_t> &expandShape) {
    int inputDim = inputShape.size();
    int expandDim = expandShape.size();
    int diff = expandDim - inputDim;

    if (diff < 0) {
        return false;
    }

    for (int i = 0; i < inputDim; ++i) {
        int inputDimSize = inputShape[inputDim - 1 - i];
        int expandDimSize = expandShape[expandDim - 1 - i];
        if (inputDimSize != expandDimSize && inputDimSize != 1) {
            return false;
        }
    }

    for (int i = 0; i < diff; ++i) {
        if (expandShape[i] <= 0) {
            return false;
        }
    }

    return true;
}

Broadcast::Broadcast(const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs, const Shape outShape,
                     const std::string &name)
    : Operator(OperatorType::BROADCAST, inputs, outputs, name, 1) {
    // Validity check
    ASSERT(inputs.size() == 1);
    ASSERT(outputs.size() <= 1);
    ASSERT(!outputs.empty() || !outShape.empty());

    if (outputs.empty()) {
        // Infer output shape and datatype
        ASSERT(isValidExpandShape(inputs[0]->tensorShape, outShape));
        Tensor *temp = new Tensor(outShape, inputs[0]->tensorDataType);
        temp->setProducer(this);
        operatorOutputs.push_back(temp);
    } else {
        ASSERT(isValidExpandShape(inputs[0]->tensorShape,
                                  outputs[0]->tensorShape));
        ASSERT(outputs[0]->tensorDataType == inputs[0]->tensorDataType);
    }

    // TODO: Tiling size and Mapping
}
} // namespace infini
