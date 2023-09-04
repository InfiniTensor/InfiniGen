# 概要设计

## 概念简介

Codegen 是一个跨加速卡平台（Nvidia GPU, Cambricon MLU, Kunlun XPU, ...）的 AI 领域代码生成器。
通过对不同硬件结构与编程模型的高度抽象，获得一个共有的、具有足够表达能力的顶层设计。
基于顶层设计，结合计算图结构与各平台编程模型特性，生成与原始图结构等价、同时高度融合的单（或者少量）核心，来获得性能上的收益。

关键字：跨平台、易拓展、代码生成、高性能

## 核心抽象

### Tensor

Tensor 既张量，多维数组。在内存上连续或者不连续，具有维度，步长，数据类型、起始偏移、常量与否等基础信息。

### Split

Split 既拆分，是一种考虑硬件结构、计算任务、张量后综合获得的数据切分规划。例如将一个一维长的数组 [0,1,2,3] 平均拆分成两份，这里的平均拆分两份既是 Split。

### Tile

Tile 既拆分后的结果，接上面的例子，[0,1,2,3] 平均拆分两份后获得 [0,1] 和 [2,3] 既是 Tile。

### Operator

Operator 既算子、运算。对整个输入张量们进行运算获得整个输出张量。例如卷积运算。

### Graph

Graph 既计算图，由张量和算子组成的复杂的图结构，静态的。

### Kernel

Kernel 既微内核，数量较少（与算子相比）的、简单的、尺寸固定的，附着在 Tile 上的运算过程。

### Core

Core 既运算核，既对于硬件最小（或者接近最小）独立运算结构的抽象。在 Cambricon MLU 上，指张量核；在 Nvidia GPU 上，指 SM。

### Memory

Memory 既存储资源，硬件结构上为多级存储结构。DRAM、Shared Memory、Register 等

## 运行过程（以单目运算 Sin 示例）

### 1. Tensor 划分 

基于硬件结构，规划出每个 Core 能够一次装下的最大尺寸，计算 Split，指导 Tensor 进行 划分，获得 Tile。

Core: 数量 4 个，单次处理能力（宽度）为 128 个数

Tensor: [0,1,2,...,1021,1022,1023]

Tile: 小于 128，既单个 Tile 的长度应当小于 Core 处理能力 128，结合运算信息与 Split 信息获得，这里不妨假定为 64。

Split: （目前概念有点模糊，这里的作用不明显）以 Tile 宽 为 64， Split 为 1024 / 64 = 16。生成 16 个 长度为 64 的 Tile。

[0,...,63], [64,...,127], ... [961,...,1024]

### 2. Tile 安排

对所有 Tile 在所有 Core 之间进行规划指定。

我们这里有 16 个 Tile, 规划给 4 个 Core，每个 Core 处理 4 个 Tile

| Core Id | Tile Id     |
| ------- | ----------- |
| Core 0  | 0,1,2,3     |
| Core 1  | 4,5,6,7     |
| Core 2  | 8,9,10,11   |
| Core 3  | 12,13,14,15 |

### 3. Codegen 生成代码

单目运算这里不涉及复杂的交叉逻辑，各个运算核心只要获取对应的数进行运算并生成，注意这里的逻辑都是编译期 Codegen，而不进行实际运算。

伪代码如下，注意到第二个例子中，我们利用了 Cache 来进行融合，避免数据的频繁搬移带来的性能下降。

这只是一个简单的例子，实际情况中我们基于整个计算图来进行代码生成。

```C++
//  y = sin(x)
int64_t core_id = getCoreID();
int64_t tile_pre_core = tile_num / core_num;
for( auto i = 0; i < tile_num; ++i) {
  int64_t tile_id = core_id * tile_pre_core + i;
  // 分配片上缓存
  Allocate_Kernel();
  // 加载数据
  Load_Kernel(Tile(tile_id));
  // 实际计算 
  Sin_Kernel(Tile(tile_id));
  // 释放片上缓存
  Free_Kernel();
  // 存储数据
  Store(Tile(tile_id));
}

//  y = sin(x); z = cos(y)
int64_t core_id = getCoreID();
int64_t tile_pre_core = tile_num / core_num;
for( auto i = 0; i < tile_num; ++i) {
  int64_t tile_id = core_id * tile_pre_core + i;
  // 分配片上缓存
  Allocate_Kernel();
  // 加载数据
  Load_Kernel(Tile(tile_id));
  // 实际计算 
  Sin_Kernel(Tile(tile_id));
  Cos_Kernel(Tile(tile_id));
  // 存储数据
  Store(Tile(tile_id));
}

```
