#!/usr/bin/env python3
"""Strict stdlib-only check; missing files and comparator errors fail the job."""
import filecmp
from pathlib import Path
import re
import sys


def verify(reference, candidate, ranks):
    sets = []
    for folder in (reference, candidate):
        files = {p.name: p for p in folder.glob('*.dat')}
        if not files:
            raise ValueError(f'no output files: {folder}')
        for rank in range(ranks):
            matches = [name for name in files if name in (
                f'Trapezoid-Adaptive1_{rank}.dat',
                f'SeriesTrapezoid-Adaptive1_{rank}.dat')]
            if len(matches) != 1:
                raise ValueError(f'missing/ambiguous final output for rank {rank}: {folder}')
        for name, path in files.items():
            if not path.stat().st_size:
                raise ValueError(f'empty output: {path}')
            with path.open() as handle:
                for line in handle:
                    if re.search(r'(?i)(?<![a-z])(nan|inf(?:inity)?)(?![a-z])', line):
                        raise ValueError(f'nonfinite output: {path}')
        sets.append(files)
    if sets[0].keys() != sets[1].keys():
        raise ValueError('output filename sets differ')
    for name in sets[0]:
        if not filecmp.cmp(sets[0][name], sets[1][name], shallow=False):
            raise ValueError(f'output differs: {name}')


if __name__ == '__main__':
    try:
        verify(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    except (ValueError, OSError) as error:
        sys.exit(f'FAIL: {error}')
    print('PASS: complete, finite, byte-identical outputs')
