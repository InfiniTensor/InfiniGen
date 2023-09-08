#include "core/api.h"

int main() {
  infini::Cache cache(256, 1024, 4, "cache01", infini::MemoryDispatch::LRU);
  infini::CacheData a = infini::CacheData("A", 0, 150);
  infini::CacheData b = infini::CacheData("B", 0, 80);
  infini::CacheData c = infini::CacheData("C", 4, 100);
  infini::CacheData d = infini::CacheData("D", 2, 25);
  infini::CacheData e = infini::CacheData("E", 4, 70);
  infini::CacheData f = infini::CacheData("F", 4, 200);

  auto res = cache.allocate(a);
  res.printInformation();
  cache.printInformation();

  res = cache.allocate(b);
  res.printInformation();
  cache.printInformation();

  res = cache.load(c);
  res.printInformation();
  cache.printInformation();

  res = cache.load(d);
  res.printInformation();
  cache.printInformation();

  res = cache.load(e);
  res.printInformation();
  cache.printInformation();

  res = cache.load(a);
  res.printInformation();
  cache.printInformation();

  res = cache.load(c);
  res.printInformation();
  cache.printInformation();

  cache.lock();

  res = cache.load(a);
  res.printInformation();
  cache.printInformation();

  res = cache.load(d);
  res.printInformation();
  cache.printInformation();

  res = cache.load(e);
  res.printInformation();
  cache.printInformation();

  cache.unlock();

  res = cache.load(b);
  res.printInformation();
  cache.printInformation();

  res = cache.free(a);
  res.printInformation();
  cache.printInformation();

  res = cache.allocate(a);
  res.printInformation();
  cache.printInformation();

  res = cache.free(b);
  res.printInformation();
  cache.printInformation();

  res = cache.allocate(f);
  res.printInformation();
  cache.printInformation();

  return 0;
}
