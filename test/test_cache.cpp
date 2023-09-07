#include "core/api.h"

int main() {
  infini::Cache cache(256, 1024, 4, "cache01", infini::MemoryDispatch::LRU);
  infini::CacheData *a = new infini::CacheData("A", 0, 150);
  infini::CacheData *b = new infini::CacheData("B", 0, 80);
  infini::CacheData *c = new infini::CacheData("C", 4, 100);
  infini::CacheData *d = new infini::CacheData("D", 2, 25);
  infini::CacheData *e = new infini::CacheData("E", 4, 70);

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

  std::vector<infini::CacheData> locked;
  locked.push_back(*a);
  cache.lock(locked);

  res = cache.load(d);
  res.printInformation();
  cache.printInformation();

  res = cache.load(e);
  res.printInformation();
  cache.printInformation();

  cache.unlock(locked);

  res = cache.load(b);
  res.printInformation();
  cache.printInformation();

  res = cache.load(a);
  res.printInformation();
  cache.printInformation();

  res = cache.free(a);
  res.printInformation();
  cache.printInformation();

  res = cache.allocate(a);
  res.printInformation();
  cache.printInformation();

  delete a;
  delete b;
  delete c;
  delete d;
  delete e;

  return 0;
}
