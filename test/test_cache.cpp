#include "core/api.h"

int main() {
    using namespace infini;
    Tensor tensor1({4, 4}, TensorDataType::CHAR);
    Tensor tensor2({8, 8}, TensorDataType::CHAR);
    tensor1.info();
    tensor2.info();
    Tiles tiles1 = tensor1.tiling({2, 2});
    tensor1.tilesInfo();
    Tiles tiles2 = tensor2.tiling({4, 4});
    tensor2.tilesInfo();

    Cache cache(32);
    cache.info();

    cache.lock();
    cache.allocate(tiles1[0]);
    cache.info();
    // tiles1[0] at [0, 4] (locked)
    cache.unlock();

    cache.load(tiles1[1]);
    cache.info();
    // tiles1[0] at [0, 4]
    // tiles1[1] at [4, 8]

    cache.allocate(tiles2[0]);
    cache.info();
    // tiles2[0] at [8, 24]

    cache.load(tiles2[1]);
    cache.info();
    // above 3 blocks are swapped out
    // tiles2[0] at [0, 16]

    cache.allocate(tiles1[1]);
    cache.info();
    // tiles2[0] at [0, 16]
    // tiles1[1] at [16, 20]

    cache.load(tiles1[1]);
    cache.info();
    // tiles2[0] at [0, 16]
    // tiles1[1] at [16, 20]

    cache.free(tiles2[1]);
    cache.info();
    // tiles1[1] at [16, 20]

    cache.allocate(tiles1[2]);
    cache.info();
    // tiles1[1] at [16, 20]
    // tiles1[2] at [20, 24]

    cache.allocate(tiles2[1]);
    cache.info();
    // tiles2[0] at [0, 16]
    // tiles1[1] at [16, 20]
    // tiles1[2] at [20, 24]

    return 0;
}
