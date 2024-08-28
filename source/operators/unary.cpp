#include "operators/unary.h"
#include "core/utils.h"

namespace infini {
Unary::Unary(const OperatorType &type, const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs, const std::string &name,
             const int64_t &outputsNum)
    : Operator(type, inputs, outputs, name, 1) {
    // Validity check
    ASSERT(inputs.size() == 1);
    ASSERT(outputs.size() <= 1);
    ASSERT(outputsNum == 1);
    if (outputs.empty()) {
        // Infer output shape and datatype
        Tensor *temp =
            new Tensor(inputs[0]->tensorShape, inputs[0]->tensorDataType);
        temp->setProducer(this);
        operatorOutputs.push_back(temp);
    } else {
        ASSERT(ALL_TRUE(outputs[0]->tensorShape == inputs[0]->tensorShape));
        ASSERT(outputs[0]->tensorDataType == inputs[0]->tensorDataType);
    }

    // TODO: Tiling size and Mapping
}
} // namespace infini
