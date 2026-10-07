# 动态版日志分析

1 GPU: all 6 full logs reached step 2834; all 6 saved comparison checks PASS; per-rank records complete.
2 GPU: all 6 full logs reached step 2834; all 6 saved comparison checks PASS; per-rank records complete.
4 GPU: all 6 full logs reached step 2834; all 6 saved comparison checks PASS; per-rank records complete.
8 GPU: all 6 full logs reached step 2834; all 6 saved comparison checks PASS; per-rank records complete.

## 当前测量结果

- 1 GPU：loop 809.697 s；逐 rank 动态启用次数 [2754]。
- 2 GPU：loop 536.428 s；逐 rank 动态启用次数 [1720, 1754]。
  - 主机阶段计时（各 rank 累计值取最大后，3 次均值）：{'plan': 8.40968, 'pack': 83.150523, 'forward': 83.177986, 'remote': 99.99082233333333, 'backward': 100.00645866666666, 'unpack': 0.006330666666666667}。
  - 正向 0.2870 GiB，返回 0.2596 GiB；单方向 2114 条消息。
- 4 GPU：loop 381.585 s；逐 rank 动态启用次数 [1720, 1754, 0, 0]。
  - 主机阶段计时（各 rank 累计值取最大后，3 次均值）：{'plan': 4.549501666666667, 'pack': 44.333052, 'forward': 59.14892966666667, 'remote': 110.51237766666667, 'backward': 90.98821166666667, 'unpack': 0.005745333333333333}。
  - 正向 0.5390 GiB，返回 0.4877 GiB；单方向 6602 条消息。
- 8 GPU：loop 304.840 s；逐 rank 动态启用次数 [580, 1300, 1754, 894, 0, 0, 0, 0]。
  - 主机阶段计时（各 rank 累计值取最大后，3 次均值）：{'plan': 2.554178, 'pack': 18.899463, 'forward': 33.718874, 'remote': 99.31196733333333, 'backward': 75.04153266666667, 'unpack': 0.004731333333333333}。
  - 正向 0.5266 GiB，返回 0.4764 GiB；单方向 19144 条消息。

## 能支持的判断

1. 2/4/8 GPU 三次运行中，动态版与同批分桶版的 DLB 输入/结果字节数及消息数逐项相同。新增动态领取没有增加这些已记录的 MPI 数据量。总 MPI 流量还包括未计入的 halo 和 collectives。
2. 8 GPU 的 rank 0–3 确实执行了动态 kernel，rank 4–7 全程旁路。此前用 MPI_MAX 后的 bypass=142 推断所有 rank 都旁路是不成立的。
3. 动态启用次数的空间差异说明各 rank 触发调度的情况不同，不能直接量化 rank 工作量差或 warp 内不平衡。旁路也不代表没有计算、没有远端任务或 Nchem 全部等于 1。
4. 已有单 GPU 对照支持 GPU 内部执行路径带来了净收益；目前无法将收益分解成求解计算、访存、任务分配或调度开销各自的贡献。

## 计时边界

forward/return 是 post+wait 的主机耗时，含等待；remote 包含远端 kernel 提交、本地 launch_local_reaction 调用及远端 stream 同步；pack 包含同步；unpack 主要是异步提交耗时。各阶段可重叠，最大值还可能来自不同 rank，不能求和或用 loop 减去它们算纯计算。

## 当前缺失的诊断

- 每 rank 本地/远端反应 kernel 的 GPU 时间，以及同一时间步内的重叠和暴露等待，需要 CUDA event 或时间线追踪。
- 当前实际任务列表中的 Nchem 分布，尤其按动态队列的实际 32 格点任务块计算的迭代离散度。历史 spatial mapping 的 warp waste 不适用于当前队列。
- 每 warp 领取块数、累计工作量、完成时间分布及最后阶段拖尾；领取块数不等本身可能是动态均衡的正常结果。
- 可用 1 - sum(nc)/(active_lanes * max(nc)) 作为块内迭代不均衡代理，尾块 padding 单独计数。该代理不是实测周期、硬件 warp efficiency 或真实反应耗时。
- 远端 kSolveReactionTasks 仍采用列表处理，也需要独立检查接收任务的分组和块内差异。

本批图只展示动态版；没有重用旧版 Nchem 直方图或虚构 GPU kernel/warp 指标。新诊断应作为独立插桩实验，正式性能继续引用现有 clean runs。

数值输出文件未包含在日志包中；正确性依据已保存的逐次 PASS 记录，本次未重新比较 .dat。
