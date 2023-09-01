#include "core/api.h"

int main() {
    infini::Cache cache(256, 512, 4, "cache01", infini::MemoryDispatch::LRU);
    infini::CacheData *a = new infini::CacheData("A", 0, 150);
    infini::CacheData *b = new infini::CacheData("B", 0, 80);
    infini::CacheData *c = new infini::CacheData("C", 4, 100);
    infini::CacheData *d = new infini::CacheData("D", 2, 25);
    infini::CacheData *e = new infini::CacheData("E", 4, 70);

    cache.loadData(a);
    cache.printInformation();
    cache.loadData(b);
    cache.printInformation();
    cache.loadData(c);
    cache.printInformation();
    cache.loadData(d);
    cache.printInformation();
    cache.loadData(e);
    cache.printInformation();
    cache.loadData(a);
    cache.printInformation();
    cache.loadData(c);
    cache.printInformation();
    return 0;
}
