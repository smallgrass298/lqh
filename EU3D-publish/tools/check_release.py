#!/usr/bin/env python3
"""Read-only integrity and saved-run checks; requires only Python's standard library."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        require(path.is_file(), f'Missing: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == entry['sha256'], f'Changed since packaging: {path}')
    print(f"PASS: {len(manifest['files'])} packaged source/result file hashes")

    recorded = ROOT / 'results/summary'
    with (recorded / 'queue_performance.csv').open(newline='') as stream:
        rows = {(int(r['gpus']), r['mode']): r for r in csv.DictReader(stream)}
    require(len(rows) == 8, 'Expected eight performance rows')
    log_count = 0
    for gpu in (1, 2, 4, 8):
        directories = list((ROOT / f'output_{gpu}/time_record/raw').glob(f'*_{gpu}gpu_suite_full'))
        require(len(directories) == 1, f'Expected one full job for {gpu} GPUs')
        folder = directories[0]
        require((folder / 'correctness.txt').read_text().startswith('PASS:'),
                f'{folder.name}: missing correctness PASS')
        require((folder / 'test_warp_queue.log').read_text().startswith('PASS:'),
                f'{folder.name}: missing queue test PASS')
        modes = ('sorted', 'dynamic') if gpu == 1 else ('all', 'dynamic_all')
        times, traffic = {}, {}
        for mode in modes:
            times[mode], traffic[mode] = [], []
            for rep in (1, 2, 3):
                path = folder / f'gpu_{mode}_rep{rep}.log'
                text = path.read_text()
                steps = re.findall(r'^GPU Iteration: (\d+)', text, re.M)
                require(steps and int(steps[-1]) == 2834, f'Incomplete: {path}')
                loops = re.findall(r'^GPU Loop time:\s*(\S+)', text, re.M)
                require(bool(loops), f'Missing loop time: {path}')
                value = float(loops[-1])
                require(math.isfinite(value) and value > 0, f'Invalid time: {path}')
                times[mode].append(value)
                check = folder / f'check_{mode}_rep{rep}.txt'
                require(check.read_text().startswith('PASS:'), f'Failed check: {check}')
                ranks = re.findall(r'GPU reaction scheduler rank=(\d+)', text)
                require(len(ranks) == gpu and set(map(int, ranks)) == set(range(gpu)),
                        f'Incomplete rank records: {path}')
                if gpu > 1:
                    match = re.search(r'GPU DLB traffic\(global cumulative\): (.*)', text)
                    require(match is not None, f'Missing traffic: {path}')
                    traffic[mode].append(dict(re.findall(r'(\w+)=(\d+)', match[1])))
                log_count += 1
            row = rows[gpu, mode]
            require(row['complete'] == 'True' and int(row['n']) == 3,
                    f'Incomplete CSV row: {gpu}/{mode}')
            for field, expected in [('mean_s', statistics.mean(times[mode])),
                                    ('stdev_s', statistics.stdev(times[mode]))]:
                require(math.isclose(float(row[field]), expected, rel_tol=1e-10, abs_tol=1e-10),
                        f'CSV mismatch: {gpu}/{mode}/{field}')
        old, new = (statistics.mean(times[m]) for m in modes)
        reduction = 100 * (old - new) / old
        require(math.isclose(float(rows[gpu, modes[1]]['time_reduction_pct']), reduction,
                             rel_tol=1e-10), f'Reduction mismatch: {gpu}')
        require(all(n < o for n, o in zip(times[modes[1]], times[modes[0]])),
                f'Paired-win claim mismatch: {gpu}')
        if gpu > 1:
            require(traffic[modes[0]] == traffic[modes[1]], f'Traffic differs: {gpu}')
        print(f'PASS: {gpu} GPU, 6 full runs, reduction={reduction:.3f}%, 3/3 wins')
    print(f'PASS: {log_count} saved full logs')
    recovered = ROOT / 'RECOVERED_DATA.json'
    if recovered.exists():
        data = json.loads(recovered.read_text())
        for entry in data['published_files']:
            path = ROOT / entry['path']
            require(path.is_file() and path.stat().st_size == entry['bytes'],
                    f'Missing or truncated recovered output: {path}')
        print(f"PASS: {len(data['published_files'])} recovered output sizes; hashes recorded at import")


if __name__ == '__main__':
    main()
