#include "core/api.h"

int main() {
    infini::Cache cache(256, 1024, 4, "cache01", infini::MemoryDispatch::LRU);
    infini::CacheData *a = new infini::CacheData("A", 0, 150);
    infini::CacheData *b = new infini::CacheData("B", 0, 80);
    infini::CacheData *c = new infini::CacheData("C", 4, 100);
    infini::CacheData *d = new infini::CacheData("D", 2, 25);
    infini::CacheData *e = new infini::CacheData("E", 4, 70);

    cache.load(a);
    cache.printInformation();
    cache.load(b);
    cache.printInformation();
    cache.load(c);
    cache.printInformation();
    cache.load(d);
    cache.printInformation();
    cache.load(e);
    cache.printInformation();
    cache.load(a);
    cache.printInformation();
    cache.load(c);
    cache.printInformation();
    cache.load(d);
    cache.printInformation();
    cache.load(e);
    cache.printInformation();
    cache.load(b);
    cache.printInformation();
    cache.load(a);
    cache.printInformation();
    return 0;
}
