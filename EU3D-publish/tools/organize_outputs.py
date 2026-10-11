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
        print(f'{output.name}: 3 timings, {len(load)} load samples per last repeat')


if __name__ == '__main__':
    main()
