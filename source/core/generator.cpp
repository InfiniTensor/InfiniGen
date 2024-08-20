#include "core/generator.h"
#include "core/tensor.h"
#include "core/tile.h"
#include "core/utils.h"
#include <algorithm>

namespace infini {

Generator::Generator(Platform platform_, Graph *graph_, Shape pattern_,
                     int64_t cacheSize)
    : platform(platform_), graph(graph_), pattern(pattern_),
      cache(Cache(cacheSize)) {
    // Preprocessing with the graph, build micro list
    std::unordered_map<Tensor *, int64_t> tempRemainingUses;
    std::vector<Operator *> sortedOps = graph->topoSort();
    registry = MicroRegistry::getInstance();

    if (pattern.empty()) {
        pattern = getProperTileShape();
    }

    for (auto data : graph->graphInputs) {
        data->tiling(pattern);
        tempRemainingUses[data] = data->tensorUsesLeft;
        params.push_back(
            std::make_pair(TO_STRING(data->tensorDataType), data->tensorName));
    }
    for (auto data : graph->graphTemps) {
        data->tiling(pattern);
        tempRemainingUses[data] = data->tensorUsesLeft;
    }
    for (auto data : graph->graphOutputs) {
        data->tiling(pattern);
        tempRemainingUses[data] = data->tensorUsesLeft;
        params.push_back(
            std::make_pair(TO_STRING(data->tensorDataType), data->tensorName));
    }

    for (int i = 0; i < sortedOps.size(); i++) {
        Micro *micro = nullptr;
        // TODO: create micro
        std::vector<Tile *> inputTiles;
        std::vector<Tile *> outputTiles;
        for (auto input : sortedOps[i]->operatorInputs) {
            inputTiles.push_back(input->tiles[0]);
        }
        for (auto output : sortedOps[i]->operatorOutputs) {
            outputTiles.push_back(output->tiles[0]);
        }
        micro = registry.getConstructor(
            MicroAttrs{sortedOps[i]->operatorType, platform.underlying()})(
            inputTiles, outputTiles);
        microList.push_back(micro);

        // Update remaining data uses
        for (auto input : sortedOps[i]->operatorInputs) {
            tempRemainingUses[input] -= 1;
            if (tempRemainingUses[input] == 0) {
                tempRemainingUses.erase(input);
            }
            // TODO: insert Free micro
            micro = registry.getConstructor(
                MicroAttrs{OperatorType::FREE, platform.underlying()})(
                {input->tiles[0]}, {});
            microList.push_back(micro);
        }

        // Store
        for (auto output : sortedOps[i]->operatorOutputs) {
            auto it = std::find(graph->graphOutputs.begin(),
                                graph->graphOutputs.end(), output);
            if (it != graph->graphOutputs.end()) {
                // TODO: insert Store micro
                micro = registry.getConstructor(
                    MicroAttrs{OperatorType::STORE, platform.underlying()})(
                    {output->tiles[0]}, {});
                microList.push_back(micro);
            }
        }
    }
}

std::string Generator::generateCode() {
    std::string code = "";
    for (auto micro : microList) {
        micro->code(cache, code);
    }
    return code;
}

Shape Generator::getProperTileShape() {
    // TODO
    return {};
}

} // namespace infini