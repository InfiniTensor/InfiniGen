#include "operators/binary.h"
#include "core/utils.h"

namespace infini {
Binary::Binary(const OperatorType &type, const std::vector<Tensor *> &inputs,
               const std::vector<Tensor *> &outputs, const std::string &name)
    : Operator(type, inputs, outputs, name, 1) {
    // Validity check
    ASSERT(inputs.size() == 2);
    ASSERT(outputs.size() <= 1);
    ASSERT(ALL_TRUE(inputs[0]->tensorShape == inputs[1]->tensorShape));
    ASSERT(inputs[0]->tensorDataType == inputs[1]->tensorDataType);
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
