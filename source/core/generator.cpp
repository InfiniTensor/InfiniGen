#include "core/generator.h"
#include "core/tensor.h"
#include "core/tile.h"
#include "core/utils.h"
#include <algorithm>
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
        tempRemainingUses[data] = data->tensorUsesLeft;
        params.push_back(dataTypeStr(data->tensorDataType) + " *" +
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
        params.push_back(dataTypeStr(data->tensorDataType) + " *" +
                         data->tensorName);
        args.push_back(data->tensorName);
    }
    code.params = STRING_GATHER(params);
    code.args = STRING_GATHER(args);
    code.dataType = STRING_SPLIT(params[0], ' ')[0];

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
              code.params + ") {\n";
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
              code.params + ") {\n";
    result += INDENTATION(indent + 1) + graph->graphName + "_device(" +
              code.args + ");\n";
    result += INDENTATION(indent) + "}\n";
    result += "\n";

    // Host Function
    result += INDENTATION(indent) + "void " + graph->graphName + "(" +
              platform.queue() + " queue, " + code.params + ") {\n";
    result += INDENTATION(indent + 1) +
              platform.taskScaleDecl(graph->graphInputs[0]->tiles) + "\n";
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

    std::regex number("(\\d+)");
    std::string expr =
        std::regex_replace(formula, number, std::string("host_src$1[i]"));

    if (platform.isBANG()) {
        // 1. Generated func decl
        args.push_back(generateHeaderFile());
        // 2. Shape
        args.push_back(INITIALIZER(graph->graphOutputs[0]->tensorShape));
        // 3. Host memory allocation
        std::string hostMemAlloc = "";
        for (auto ptr : hostPointers) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc(LEN * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, ptr);
        }
        args.push_back(hostMemAlloc);
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit += fmt::format("{0}{1}[i] = distrib(engine);\n",
                                       INDENTATION(4), hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto ptr : devicePointers) {
            deviceMemAlloc +=
                fmt::format("{0}CNRT_CHECK(cnrtMalloc((void **)&{1}, LEN * "
                            "sizeof({2})));\n",
                            INDENTATION(2), ptr, code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit +=
                fmt::format("{0}CNRT_CHECK(cnrtMemcpy({1}, {2}, LEN * "
                            "sizeof({3}), cnrtMemcpyHostToDev));\n",
                            INDENTATION(2), devicePointers[i], hostPointers[i],
                            code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H = fmt::format(
            "CNRT_CHECK(cnrtMemcpy({0}, {1}, LEN * sizeof({2}), "
            "cnrtMemcpyDevToHost));",
            hostPointers.back(), devicePointers.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 11. Free pointers
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
        // 2. Shape
        args.push_back(INITIALIZER(graph->graphOutputs[0]->tensorShape));
        // 3. Host memory allocation
        std::string hostMemAlloc = "";
        for (auto ptr : hostPointers) {
            hostMemAlloc +=
                fmt::format("{0}{1} *{2} = ({1}*)malloc(LEN * sizeof({1}));\n",
                            INDENTATION(2), code.dataType, ptr);
        }
        args.push_back(hostMemAlloc);
        // 4. Host memory initialization
        std::string hostMemInit = "";
        for (auto i = 0; i < hostPointers.size() - 1; i++) {
            hostMemInit += fmt::format("{0}{1}[i] = distrib(engine);\n",
                                       INDENTATION(4), hostPointers[i]);
        }
        args.push_back(hostMemInit);
        // 5. Device memory allocation
        std::string deviceMemAlloc = "";
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}{1} *{2};\n", INDENTATION(2),
                                          code.dataType, ptr);
        }
        for (auto ptr : devicePointers) {
            deviceMemAlloc += fmt::format("{0}cudaMalloc((void **)&{1}, LEN * "
                                          "sizeof({2}));\n",
                                          INDENTATION(2), ptr, code.dataType);
        }
        args.push_back(deviceMemAlloc);
        // 6. Device memory initialization
        std::string deviceMemInit = "";
        for (auto i = 0; i < devicePointers.size() - 1; i++) {
            deviceMemInit +=
                fmt::format("{0}cudaMemcpy({1}, {2}, LEN * "
                            "sizeof({3}), cudaMemcpyHostToDevice);\n",
                            INDENTATION(2), devicePointers[i], hostPointers[i],
                            code.dataType);
        }
        args.push_back(deviceMemInit);
        // 7 & 8. Warmup and Execute
        std::string exec = fmt::format("{0}(queue, {1});", graph->graphName,
                                       STRING_GATHER(devicePointers));
        args.push_back(exec);
        args.push_back(exec);
        // 9. Copy result to host
        std::string resD2H = fmt::format(
            "cudaMemcpy({0}, {1}, LEN * sizeof({2}), "
            "cudaMemcpyDeviceToHost);",
            hostPointers.back(), devicePointers.back(), code.dataType);
        args.push_back(resD2H);
        // 10. Calculate baseline
        std::string calc = fmt::format("{0} res = {1};", code.dataType, expr);
        args.push_back(calc);
        // 11. Free pointers
        std::string freePtrs;
        for (auto ptr : devicePointers) {
            freePtrs += fmt::format("{0}cudaFree({1});\n", INDENTATION(2), ptr);
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
