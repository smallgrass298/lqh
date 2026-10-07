"""Extract current-run load and timing records into output_<ranks> directories."""
import csv
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def write_table(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    for gpu in (1, 2, 4, 8):
        output = ROOT / f'output_{gpu}'
        for name in ('grid', 'load', 'results', 'time_record'):
            (output / name).mkdir(parents=True, exist_ok=True)
        job, = (output / 'time_record/raw').glob('*_suite_full')
        mode = 'dynamic' if gpu == 1 else 'dynamic_all'
        timing = []
        for repeat in (1, 2, 3):
            log = job / f'gpu_{mode}_rep{repeat}.log'
            text = log.read_text()
            steps = re.findall(r'^GPU Iteration: (\d+)', text, re.M)
            assert steps and int(steps[-1]) == 2834
            values = re.findall(r'^GPU Loop time:\s*(\S+)', text, re.M)
            assert len(values) == 1
            timing.append(dict(repeat=repeat, mode=mode, steps=2834,
                               loop_s=values[0], source=log.relative_to(output).as_posix()))
            load = []
            pending = None
            for line in text.splitlines():
                match = re.match(r'Load Degree:\s*(\S+)\s+NchemMax:\s*(\d+)', line)
                if match:
                    assert pending is None
                    pending = match.groups()
                iteration = re.match(r'GPU Iteration:\s*(\d+)', line)
                if pending and iteration:
                    load.append(dict(iteration=iteration[1], load_degree=pending[0],
                                     nchem_max=pending[1]))
                    pending = None
            assert pending is None
            if load:
                write_table(output / f'load/from_log_rep{repeat}.tsv', load,
                            ['iteration', 'load_degree', 'nchem_max'])
            check = job / f'check_{mode}_rep{repeat}.txt'
            assert check.read_text().startswith('PASS:')
            shutil.copyfile(check, output / f'results/check_rep{repeat}.txt')
        write_table(output / 'time_record/loop_times.tsv', timing,
                    ['repeat', 'mode', 'steps', 'loop_s', 'source'])
        shutil.copyfile(ROOT / 'solver/input/grid-3dRSBI-nk32.txt',
                        output / 'grid/input_grid_parameters.txt')
        (output / 'grid/README.md').write_text(
            '# 网格\n\ninput_grid_parameters.txt 是本次运行使用的网格输入参数副本。'
            '运行时生成的 grid_check_<rank>.dat 未包含在本地日志包中。\n')
        missing = (
            '# 数值结果\n\n当前动态模式的三次输出检查记录为 check_rep1/2/3.txt。'
            '它们是当时保存的逐字节比较 PASS 记录，不是数值场。\n\n'
            f'完整数值场缺失。需要从集群原作业 {job.name} 的 '
            f'results_{mode}_rep1/2/3 目录取回 .dat，分别放入本目录的 rep1/2/3。'
            '如果集群副本已不存在，则需要重新运行，不能从日志恢复数值场。\n')
        recovered = (
            '# 数值结果\n\n本目录的 .dat 为当前动态模式第一次完整运行的输出，'
            f'来源：{job.name}/results_{mode}_rep1。文件编号 0/1 表示两次输出，'
            '末尾编号表示 MPI rank。文件头包含 X/Y/Z 坐标和流场变量。\n\n'
            'check_rep1/2/3.txt 为原作业的检查记录。恢复后已验证三次动态结果与'
            '同规模静态参考 SHA-256 一致，并检查 NaN/Inf。详见根目录 RECOVERED_DATA.json。\n\n'
            '本地使用硬链接共享恢复目录的数据，不增加一份数据副本。请勿原地编辑这些文件。'
            '大型 .dat 已由 Git 忽略，需要单独上传数据附件。\n')
        (output / 'results/README.md').write_text(
            recovered if list((output / 'results').glob('*.dat')) else missing)
        (output / 'load/README.md').write_text(
            '# 负载记录\n\n' + (
                '单 rank 日志没有 Load Degree 记录，不能将缺失值当作零。\n'
                if gpu == 1 else
                'from_log_rep1/2/3.tsv 从动态模式日志提取迭代号、Load Degree 和 NchemMax。'
                '迭代号取自负载行后的 GPU Iteration 行，与 Euler_GPU.cpp 输出顺序一致。'
                '这些是日志精度的重建记录，不是原始 LoadDegree.dat。\n'))
        (output / 'time_record/README.md').write_text(
            '# 计时记录\n\nloop_times.tsv 是当前动态模式三次完整运行的 GPU loop 耗时。'
            'source 列指向原始日志。raw/ 保留本规模完整实验和短测的原始日志，'
            '包括同批静态对照、构建日志和检查记录。未保存的 CPU 风格分阶段计时文件不补造。\n')
        print(f'{output.name}: 3 timings, {len(load)} load samples per last repeat')


if __name__ == '__main__':
    main()
