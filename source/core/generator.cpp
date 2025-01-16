#include "core/generator.h"
#include "core/tensor.h"
#include "core/tile.h"
#include "core/utils.h"
#include <algorithm>
#include <deque>
#ifdef DEBUG_MODE
#include <fmt/core.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#endif

namespace infini {

Generator::Generator(Platform platform_, Graph *graph_, Shape pattern_,
                     int64_t cacheSize)
    : platform(platform_), graph(graph_), pattern(pattern_),
      cache(Cache(cacheSize)) {
    // Preprocess graph
    std::unordered_map<Tensor *, int64_t> tempRemainingUses;
    std::vector<Operator *> sortedOps = graph->topoSort();
    registry = MicroRegistry::getInstance();

    if (pattern.empty()) {
        pattern = getProperTileShape();
    } else {
        // TODO: check pattern valid
    }

    std::vector<std::string> args;
    std::vector<std::string> params;
    for (auto data : graph->graphInputs) {
        data->tiling(pattern);
        data->tiles[0]->tileCoordsExpr =
            data->tiles[0]->tileId2TileCoords(platform.taskId());
        tempRemainingUses[data] = data->tensorUsesLeft;
        params.push_back(dataTypeStr(data->tensorDataType) + "* " +
                         data->tensorName);
        args.push_back(data->tensorName);
    }
    for (auto data : graph->graphTemps) {
        data->tiling(pattern);
        tempRemainingUses[data] = data->tensorUsesLeft;
    }
    for (auto data : graph->graphOutputs) {
        data->tiling(pattern);
        tempRemainingUses[data] = data->tensorUsesLeft;
        params.push_back(dataTypeStr(data->tensorDataType) + "* " +
                         data->tensorName);
        args.push_back(data->tensorName);
    }
    code.params = STRING_GATHER(params);
    code.args = STRING_GATHER(args);
    code.dataType = STRING_SPLIT(params[0], '*')[0];
    if (platform == Platform::ASCEND) {
        std::vector<std::string> paramsOnChip;
        std::transform(params.begin(), params.end(),
                       std::back_inserter(paramsOnChip),
                       [](const std::string &s) { return "__gm__ " + s; });
        code.paramsOnChip = STRING_GATHER(paramsOnChip);
    } else if (platform == Platform::KUNLUN) {
        std::vector<std::string> paramsOnChip;
        std::transform(params.begin(), params.end(),
                        std::back_inserter(paramsOnChip),
                        [](const std::string &s) { return "_global_ptr_ " + s; });
        code.paramsOnChip = STRING_GATHER(paramsOnChip);
    } else {
        code.paramsOnChip = code.params;
    }

    // TODO
    auto currentCoords = graph->graphInputs[0]->tiles[0]->tileCoordsExpr;
    // Apply mapping
    for (auto op : graph->graphOperators) {
        if (op->operatorType == OperatorType::BROADCAST) {
            op->operatorOutputs[0]->tiles[0]->tileCoordsExpr = currentCoords;
            std::vector<std::string> inputTileCoords;
            for (int i = 0; i < currentCoords.size(); i++) {
                inputTileCoords.push_back(
                    "(" + currentCoords[i] + " % " +
                    std::to_string(op->operatorInputs[0]->tileGridShape[i]) +
                    ")");
            }
            std::deque<Operator *> previousOps(op->operatorPredecessors.begin(),
                                               op->operatorPredecessors.end());
            while (!previousOps.empty()) {
                auto ptr = previousOps.front();
                for (auto input : ptr->operatorInputs) {
                    input->tiles[0]->tileCoordsExpr = inputTileCoords;
                }
                previousOps.insert(previousOps.end(),
                                   ptr->operatorPredecessors.begin(),
                                   ptr->operatorPredecessors.end());
                previousOps.pop_front();
            }
        } else {
            for (auto output : op->operatorOutputs) {
                output->tiles[0]->tileCoordsExpr = currentCoords;
            }
        }
    }

    // Build micro list
    for (int i = 0; i < sortedOps.size(); i++) {
        Micro *micro = nullptr;
        // Create micro
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
            // Insert Free micro
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
                // Insert Store micro
                micro = registry.getConstructor(
                    MicroAttrs{OperatorType::STORE, platform.underlying()})(
                    {output->tiles[0]}, {});
                microList.push_back(micro);
            }
        }
    }
}

Generator::~Generator() {
    for (auto micro : microList) {
        delete micro;
    }
}

std::string Generator::generateHeaderFile(const std::string &filepath,
                                          const int64_t &indent) {
    std::string result = INDENTATION(indent) + "void " + graph->graphName +
                         "(" + platform.queue() + " queue, " + code.params +
                         ");\n";
    if (filepath != "") {
        LOG_FILE(filepath) << result;
    }
    return result;
}

std::string Generator::generateSourceFile(const std::string &filepath,
                                          const int64_t &indent) {
    std::string result = INDENTATION(indent);
    // Header
    result += platform.head();
    result += "\n";

    // Device Function
    // TODO: multiple tasks
    result += INDENTATION(indent) +
              platform.deviceFuncDecl(graph->graphName + "_device") + "(" +
              code.paramsOnChip + ") {\n";
    result +=
        INDENTATION(indent + 1) +
        platform.cacheDecl(cache.cacheName, cache.cacheSize, code.dataType) +
        "\n";
    for (auto micro : microList) {
        micro->code(cache, result, indent + 1);
    }
    result += INDENTATION(indent) + "}\n";
    result += "\n";

    // Global Function
    result += INDENTATION(indent) +
              platform.globalFuncDecl(graph->graphName + "_global") + "(" +
              code.paramsOnChip + ") {\n";
    result += INDENTATION(indent + 1) + graph->graphName + "_device(" +
              code.args + ");\n";
    result += INDENTATION(indent) + "}\n";
    result += "\n";

    // Host Function
    result += INDENTATION(indent) + "void " + graph->graphName + "(" +
              platform.queue() + " queue, " + code.params + ") {\n";
    result += INDENTATION(indent + 1) +
              platform.taskScaleDecl(graph->graphOutputs[0]->tiles) + "\n";
    result += INDENTATION(indent + 1) + graph->graphName + "_global" +
              platform.syntacticSugar() + "(" + code.args + ");\n";
    result += INDENTATION(indent) + "}\n";
    result += "\n";

    if (filepath != "") {
        LOG_FILE(filepath) << result;
    }
    return result;
}

Shape Generator::getProperTileShape() {
    // TODO
    return {};
}

#ifdef DEBUG_MODE
std::string Generator::generateTestScript(const std::string &templateFilepath,
                                          const std::string &formula,
                                          const std::string &filepath) {
    std::string fileContent = "";

    std::ifstream infile(templateFilepath);
    if (!infile.is_open()) {
        LOG(ERROR) << "Template file does not exsit: " << templateFilepath;
        return "";
    }

    std::stringstream buffer;
    buffer << infile.rdbuf();
    fileContent = buffer.str();

    std::vector<std::string> args;

    std::vector<int64_t> tensorLengths;
    for (auto input : graph->graphInputs) {
        tensorLengths.push_back(input->getElementNum());
    }
    tensorLengths.push_back(graph->graphOutputs[0]->getElementNum());

    std::vector<std::string> hostPointers;
    for (auto i = 0; i < graph->graphInputs.size(); i++) {
        hostPointers.push_back(fmt::format("host_src{}", i));
    }
    hostPointers.push_back("host_dest");
    std::vector<std::string> devicePointers;
    for (auto i = 0; i < graph->graphInputs.size(); i++) {
        devicePointers.push_back(fmt::format("dev_src{}", i));
    }
    devicePointers.push_back("dev_dest");

    std::regex number("(\\d+)\\[");
    std::string expr =
        std::regex_replace(formula, number, std::string("host_src$1["));

    if (platform.isBANG()) {
        // 1. Generated func decl
        args.push_back(generateHeaderFile());
        // 2. Host memory allocation
        std::string hostMemAlloc = "";
        for (int i = 0; i < hostPointers.size(); i++) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc({3} * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, hostPointers[i],
                            tensorLengths[i]);
        }
        args.push_back(hostMemAlloc);
        // 3. Maximum tensor length
        args.push_back(std::to_string(
            *(std::max_element(tensorLengths.begin(), tensorLengths.end()))));
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit +=
                fmt::format("{0}if (i < {1}) {2}[i] = distrib(engine);\n",
                            INDENTATION(4), tensorLengths[i], hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto i = 0; i < devicePointers.size(); i++) {
            deviceMemAlloc +=
                fmt::format("{0}CNRT_CHECK(cnrtMalloc((void **)&{1}, {2} * "
                            "sizeof({3})));\n",
                            INDENTATION(2), devicePointers[i], tensorLengths[i],
                            code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit +=
                fmt::format("{0}CNRT_CHECK(cnrtMemcpy({1}, {2}, {3} * "
                            "sizeof({4}), cnrtMemcpyHostToDev));\n",
                            INDENTATION(2), devicePointers[i], hostPointers[i],
                            tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H =
            fmt::format("CNRT_CHECK(cnrtMemcpy({0}, {1}, {2} * sizeof({3}), "
                        "cnrtMemcpyDevToHost));",
                        hostPointers.back(), devicePointers.back(),
                        tensorLengths.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Output tensor length
        args.push_back(std::to_string(tensorLengths.back()));
        // 11. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 12. Free pointers
        std::string freePtrs;
        for (auto ptr : devicePointers) {
            freePtrs += fmt::format("{0}cnrtFree({1});\n", INDENTATION(2), ptr);
        }
        for (auto ptr : hostPointers) {
            freePtrs += fmt::format("{0}free({1});\n", INDENTATION(2), ptr);
        }
        args.push_back(freePtrs);

    } else if (platform.isCUDA()) {
        // 1. Generated func decl
        args.push_back(generateHeaderFile());
        // 2. Host memory allocation
        std::string hostMemAlloc = "";
        for (int i = 0; i < hostPointers.size(); i++) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc({3} * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, hostPointers[i],
                            tensorLengths[i]);
        }
        args.push_back(hostMemAlloc);
        // 3. Maximum tensor length
        args.push_back(std::to_string(
            *(std::max_element(tensorLengths.begin(), tensorLengths.end()))));
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit +=
                fmt::format("{0}if (i < {1}) {2}[i] = distrib(engine);\n",
                            INDENTATION(4), tensorLengths[i], hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto i = 0; i < devicePointers.size(); i++) {
            deviceMemAlloc += fmt::format("{0}cudaMalloc((void **)&{1}, {2} * "
                                          "sizeof({3}));\n",
                                          INDENTATION(2), devicePointers[i],
                                          tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit +=
                fmt::format("{0}cudaMemcpy({1}, {2}, {3} * "
                            "sizeof({4}), cudaMemcpyHostToDevice);\n",
                            INDENTATION(2), devicePointers[i], hostPointers[i],
                            tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H =
            fmt::format("cudaMemcpy({0}, {1}, {2} * sizeof({3}), "
                        "cudaMemcpyDeviceToHost);",
                        hostPointers.back(), devicePointers.back(),
                        tensorLengths.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Output tensor length
        args.push_back(std::to_string(tensorLengths.back()));
        // 11. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 12. Free pointers
        std::string freePtrs;
        for (auto ptr : devicePointers) {
            freePtrs += fmt::format("{0}cudaFree({1});\n", INDENTATION(2), ptr);
        }
        for (auto ptr : hostPointers) {
            freePtrs += fmt::format("{0}free({1});\n", INDENTATION(2), ptr);
        }
        args.push_back(freePtrs);

    } else if (platform.isASCEND()) {
        // 1. Generated func decl
        args.push_back(generateHeaderFile());
        // 2. Host memory allocation
        std::string hostMemAlloc = "";
        for (int i = 0; i < hostPointers.size(); i++) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc({3} * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, hostPointers[i],
                            tensorLengths[i]);
        }
        args.push_back(hostMemAlloc);
        // 3. Maximum tensor length
        args.push_back(std::to_string(
            *(std::max_element(tensorLengths.begin(), tensorLengths.end()))));
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit +=
                fmt::format("{0}if (i < {1}) {2}[i] = distrib(engine);\n",
                            INDENTATION(4), tensorLengths[i], hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto i = 0; i < devicePointers.size(); i++) {
            deviceMemAlloc +=
                fmt::format("{0}CHECK_ACL(aclrtMalloc((void **)&{1}, {2} * "
                            "sizeof({3}), ACL_MEM_MALLOC_HUGE_FIRST));\n",
                            INDENTATION(2), devicePointers[i], tensorLengths[i],
                            code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit += fmt::format(
                "{0}CHECK_ACL(aclrtMemcpy({1}, {3} * sizeof({4}), {2}, {3} * "
                "sizeof({4}), ACL_MEMCPY_HOST_TO_DEVICE));\n",
                INDENTATION(2), devicePointers[i], hostPointers[i],
                tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H =
            fmt::format("CHECK_ACL(aclrtMemcpy({0}, {2} * sizeof({3}), {1}, "
                        "{2} * sizeof({3}), "
                        "ACL_MEMCPY_DEVICE_TO_HOST));",
                        hostPointers.back(), devicePointers.back(),
                        tensorLengths.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Output tensor length
        args.push_back(std::to_string(tensorLengths.back()));
        // 11. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 12. Free pointers
        std::string freePtrs;
        for (auto ptr : devicePointers) {
            freePtrs +=
                fmt::format("{0}aclrtFree({1});\n", INDENTATION(2), ptr);
        }
        for (auto ptr : hostPointers) {
            freePtrs += fmt::format("{0}free({1});\n", INDENTATION(2), ptr);
        }
        args.push_back(freePtrs);

    } else if (platform.isKUNLUN()) {
        // 1. Generated func decl
        args.push_back(generateHeaderFile());
        // 2. Host memory allocation
        std::string hostMemAlloc = "";
        for (int i = 0; i < hostPointers.size(); i++) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc({3} * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, hostPointers[i],
                            tensorLengths[i]);
        }
        args.push_back(hostMemAlloc);
        // 3. Maximum tensor length
        args.push_back(std::to_string(
            *(std::max_element(tensorLengths.begin(), tensorLengths.end()))));
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit +=
                fmt::format("{0}if (i < {1}) {2}[i] = distrib(engine);\n",
                            INDENTATION(4), tensorLengths[i], hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto i = 0; i < devicePointers.size(); i++) {
            deviceMemAlloc += fmt::format("{0}xpu_malloc((void **)&{1}, {2} * "
                                          "sizeof({3}));\n",
                                          INDENTATION(2), devicePointers[i],
                                          tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit +=
                fmt::format("{0}xpu_memcpy({1}, {2}, {3} * "
                            "sizeof({4}), XPU_HOST_TO_DEVICE));\n",
                            INDENTATION(2), devicePointers[i], hostPointers[i],
                            tensorLengths[i], code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H =
            fmt::format("xpu_memcpy({0}, {1}, {2} * sizeof({3}), "
                        "XPU_DEVICE_TO_HOST));",
                        hostPointers.back(), devicePointers.back(),
                        tensorLengths.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Output tensor length
        args.push_back(std::to_string(tensorLengths.back()));
        // 11. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 12. Free pointers
        std::string freePtrs;
        for (auto ptr : devicePointers) {
            freePtrs +=
                fmt::format("{0}xpu_free({1});\n", INDENTATION(2), ptr);
        }
        for (auto ptr : hostPointers) {
            freePtrs += fmt::format("{0}free({1});\n", INDENTATION(2), ptr);
        }
        args.push_back(freePtrs);

    } else {
        LOG(ERROR) << "Platform not supported now.";
        return "";
    }

    size_t pos = 0;
    size_t index = 0;

    // Loop through each placeholder {}
    while ((pos = fileContent.find("{}", pos)) != std::string::npos &&
           index < args.size()) {
        fileContent.replace(pos, 2, args[index++]);
        pos += args[index - 1].length(); // Move past the replaced content
    }

    if (filepath != "") {
        LOG_FILE(filepath) << fileContent;
    }

    return fileContent;
}
#endif

} // namespace infini
