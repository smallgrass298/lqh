# 方法

## 工作脉络

CPU/MPI 求解器迁移到 CUDA 后，先加入跨 rank 反应任务迁移与 CUDA-aware MPI，再加入通信重叠、成本门控和 Nchem 分桶。当前版本在分桶基础上加入动态 warp 队列。

## 动态队列

本地反应格点按 Nchem 分成 8 个桶，排除已迁出的格点。warp 从重桶开始，由 lane 0 用 `atomicAdd(..., 32)` 领取最多 32 个格点，再广播给其他 lane。每条有效 lane 独立求解一个格点。

完成当前块并同步后，warp 继续领取。当前桶没有未领取任务时可进入下一桶，不等待其他 warp 完成已领取任务。块不跨桶，尾块的无效 lane 仍参与同步。

worker 数由 kernel occupancy 查询、SM 数和输入规模决定，每次反应前重置队列。负载较均匀时使用普通 kernel；`EU3D_REACTION_SCHED_MIN_HEAVY=0` 可强制测试队列。

队列只改变本地未迁移格点的调度，远端反应和 MPI 迁移协议沿用原实现。

## 模式

| 配置 | 单 GPU | 多 GPU |
|---|---|---|
| 当前动态队列 | `dynamic` | `dynamic_all` |
| 同批静态分桶对照 | `sorted` | `all` |

多 GPU 两种模式都启用通信重叠、成本门控和迁移任务排序，仅本地调度不同。开关由 `solver/run_intragpu_common.sh` 设置；其他消融模式保留。

## 源码入口

| 文件 | 作用 |
|---|---|
| `solver/cpu_src/Main.cpp` | 程序入口 |
| `solver/cpu_src/Euler_GPU.cpp` | CPU 初始化、输出与 GPU 驱动 |
| `solver/src/gpu_solver.cu` | 显存管理与时间步进 |
| `solver/src/reaction.cu` | 反应、分桶、队列与 DLB |
| `solver/src/warp_queue.cuh` | warp 原子领取 |
| `solver/src/dlb_schedule.hpp` | 跨 rank 计划与成本门控 |
| `solver/src/test_warp_queue.cu` | 队列边界与 exact-once 测试 |

动态领取允许先完成的 warp 继续处理未分配任务。现有实验测量整体耗时，尚不能分离计算、访存和调度开销的贡献。
