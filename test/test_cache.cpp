#include "core/api.h"

int main() {
    using namespace infini;
    Tensor tensor1({16, 32}, TensorDataType::FLOAT);
    Tensor tensor2({128, 64}, TensorDataType::FLOAT);
    tensor1.info();
    tensor2.info();
    Tiles tiles1 = tensor1.tiling({16, 16});
    tensor1.tilesInfo();
    Tiles tiles2 = tensor2.tiling({32, 32});
    tensor2.tilesInfo();

    Cache cache(4096 * 4);
    cache.info();

    Block *b1 = cache.allocate(tiles1[0]);
    cache.info();

    Block *b2 = cache.allocate(tiles1[1]);
    cache.info();

    cache.free(tiles1[0]);
    Block *b3 = cache.allocate(tiles2[0]);
    cache.info();

    cache.free(tiles1[1]);
    Block *b4 = cache.allocate(tiles2[1]);
    cache.info();

    cache.free(tiles2[0]);
    cache.info();

    return 0;
}
