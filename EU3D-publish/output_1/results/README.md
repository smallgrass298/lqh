# 数值结果

本目录的 .dat 为当前动态模式第一次完整运行的输出，来源：4023062.pbs-7_1gpu_suite_full/results_dynamic_rep1。文件编号 0/1 表示两次输出，末尾编号表示 MPI rank。文件头包含 X/Y/Z 坐标和流场变量。

check_rep1/2/3.txt 为原作业的检查记录。恢复后已验证三次动态结果与同规模静态参考 SHA-256 一致，并检查 NaN/Inf。详见根目录 RECOVERED_DATA.json。

本地使用硬链接共享恢复目录的数据，不增加一份数据副本。请勿原地编辑这些文件。大型 .dat 已由 Git 忽略，需要单独上传数据附件。
