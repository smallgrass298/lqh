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

## 计时边界

forward/return 是 post+wait 的主机耗时，含等待；remote 包含远端 kernel 提交、本地 launch_local_reaction 调用及远端 stream 同步；pack 包含同步；unpack 主要是异步提交耗时。各阶段可重叠，最大值还可能来自不同 rank。

