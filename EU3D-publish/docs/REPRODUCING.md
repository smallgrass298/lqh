# 构建、检查与复现

所有命令从整理包根目录执行，除非命令中显式 `cd solver`。

## 无 GPU 的本地检查

```bash
python3 tools/check_release.py
cd solver
python3 -B -m unittest test_verify_queue_outputs.py
c++ -std=c++17 -O2 -I./src src/test_dlb_schedule.cpp -o /tmp/eu3d_test_dlb_schedule
/tmp/eu3d_test_dlb_schedule
```

检查文件完整性、实验日志和 CPU 端调度逻辑。

## 构建环境

当前 PBS 脚本针对 Imperial CX3 L40S，自动加载 `tools/prod`、`CUDA/12.6.0`、`OpenMPI/5.0.3-GCC-13.3.0`、`UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0`。在其他集群上，应根据实际设备调整 PBS 资源、module 名称、MPI/UCX 设置和架构参数。

在已配置好 CUDA/MPI 的环境中：

```bash
cd solver
make -j4 ARCH=-arch=sm_89 NUMERIC_MODE=native
make test_dlb_schedule test_warp_queue
./test_dlb_schedule
./test_warp_queue
```

实验使用 `NUMERIC_MODE=native`，Makefile 默认值为 `precise`。切换编译参数前执行 `make clean`。

## CX3 上运行正式实验

```bash
cd solver
bash submit_queue_all.sh
```

提交 1、2、4、8 GPU 四个 PBS 作业。每个作业先构建、运行调度测试，再进行 100 步短测，通过后运行 2834 步完整实验，每模式 3 次。1 GPU 短测还执行队列单测的 memcheck 与 synccheck。

单独提交某个规模可用 `qsub run_queue_suite_1gpu.pbs`，其他规模为 2、4、8。已有 `run_queue_smoke1/2.pbs` 和 `run_queue_full1/2.pbs` 保留作分步实验入口。`run_intragpu_common.sh` 由 PBS 作业调用。

作业输出保存在 `solver/intragpu_results/<jobid>_<tag>/`，包含构建日志、GPU 日志和结果检查记录。

运行结束后汇总新作业：

```bash
cd solver
python3 summarize_queue_suite.py intragpu_results
```

## 复核已保存结果

```bash
python3 tools/check_release.py
```

该命令重新读取 24 个完整日志，核对步数、计时、逐次 PASS、rank 记录、均值/样本标准差/下降比例，并检查旧版与动态版的 DLB 流量一致性。它还检查整理时保存的文件 SHA-256。

若要使用原汇总器重新生成 CSV，请先复制到新目录，保留打包时的原始汇总：

```bash
mkdir -p results/generated
mkdir -p results/generated/recomputed
cp -R output_*/time_record/raw/* results/generated/recomputed/
python3 solver/summarize_queue_suite.py results/generated/recomputed
```

## 重绘图表

```bash
python3 -m venv .venv
.venv/bin/pip install -r analysis/requirements.txt
.venv/bin/python analysis/analyze_current_warpqueue.py
.venv/bin/python analysis/plot_current_performance.py
```

日志分析写入 `results/generated/current_analysis/`，当前性能图写入 `results/figures/current_performance.png`。两个脚本均读取最后一批实验记录。
