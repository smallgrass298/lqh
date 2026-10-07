# 文件来源与输出

代码、输入和最后一批日志来自 `EU3D-cuda-dlb-warpqueue`。原项目未修改。

2026-10-04 移除历史对比图及其脚本，只保留当前动态版图表。同批静态对照日志用于复核性能提升，仍保留。代码注释改为简洁英文，计算逻辑和输入不变；Python 输出检查器的一条成功提示改为“误差满足指定阈值”。

`SOURCE_MANIFEST.json` 记录原文件路径及整理前后 SHA-256。新增文档、检查工具和当前性能绘图程序不属于原文件副本。

## 运行输出

`output_2` 表示使用 2 个 MPI 进程运行，不表示第二版；`output_8` 同理。

| 子目录 | 内容 |
|---|---|
| `grid/` | 各 rank 网格检查文件 |
| `load/` | 负载与不平衡度 |
| `results/` | 数值场 `.dat` |
| `time_record/` | 计算、总耗时与分阶段计时 |

CPU/GPU 共用目录创建逻辑，实际写入的计时文件取决于执行路径。PBS 在临时目录运行，将日志与结果快照复制到 `solver/intragpu_results/<jobid>_<tag>/`。

本仓库顶层 `output_1/2/4/8` 按上述四类整理。原始日志在各自的 `time_record/raw/`；跨规模汇总在 `results/summary/`。`tools/organize_outputs.py` 从日志提取当前动态模式的计时与负载。

原日志包未包含数值场；2026-10-07 从集群恢复包取回 120 个 .dat。各规模 `results/` 选取动态模式第一次完整运行，共 30 个文件。三次重复与同规模静态参考逐文件哈希一致，完整来源清单见 `RECOVERED_DATA.json`。

本地选定文件通过硬链接引用恢复目录中的数据，无额外数值场副本。删除一个路径不会删除另一个路径的数据；原地编辑则会影响两处，因此请将数值场作为只读数据使用。独立网格检查文件仍未取回；数值场文件头包含 X/Y/Z 坐标。

大型 .dat 由 `.gitignore` 排除，需要单独上传数据附件。尚未上传或删除本地恢复包，只有确认远端数据完整后才清理本地数据。

`obj/`、可执行程序及 `solver/output_*` 运行输出由 Git 忽略。顶层整理后的 `output_*` 应提交。

## 推送

在本目录执行：

```bash
git init -b main
git add .
git status --short
git commit -m "Add EU3D dynamic warp queue solver and results"
git remote add origin <仓库地址>
git push -u origin main
```

尚未执行推送。未添加许可证，沿用原代码归属。
