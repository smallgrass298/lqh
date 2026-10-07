"""Verify recovered repeats and hard-link one current run per GPU count."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    digest = hashlib.sha256()
    tail = b''
    with path.open('rb') as stream:
        header = stream.readline()
        if not header.startswith(b'Variables = X,Y,Z,'):
            raise ValueError(f'Unexpected header: {path}')
        stream.seek(0)
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
            data = tail + block
            if re.search(rb'(?i)(?<![a-z])(nan|inf(?:inity)?)(?![a-z])', data):
                raise ValueError(f'Nonfinite value: {path}')
            tail = data[-32:]
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    records, selected = [], []
    for gpu, job in ((1, 4023062), (2, 4023063), (4, 4023064), (8, 4023065)):
        mode = 'dynamic' if gpu == 1 else 'dynamic_all'
        baseline = 'sorted' if gpu == 1 else 'all'
        folder = source / 'intragpu_results' / f'{job}.pbs-7_{gpu}gpu_suite_full'
        reference = None
        for name in [f'results_{baseline}_rep1',
                     *[f'results_{mode}_rep{r}' for r in (1, 2, 3)]]:
            paths = sorted((folder / name).glob('*.dat'))
            prefix = 'Series' if gpu == 1 else ''
            expected = {f'{prefix}Trapezoid-Adaptive{s}_{rank}.dat'
                        for s in (0, 1) for rank in range(gpu)}
            if {p.name for p in paths} != expected:
                raise ValueError(f'Incomplete output set: {folder / name}')
            hashes = {}
            for p in paths:
                hashes[p.name] = inspect(p)
                records.append(dict(source=p.relative_to(source).as_posix(),
                                    bytes=p.stat().st_size, sha256=hashes[p.name]))
            if reference is None:
                reference = hashes
            elif hashes != reference:
                raise ValueError(f'Output mismatch: {folder / name}')
            if name == f'results_{mode}_rep1':
                selected.extend((gpu, p, hashes[p.name]) for p in paths)
            print(f'PASS: {gpu} GPU {name}, {len(paths)} files', flush=True)
    destinations = [(p, ROOT / f'output_{gpu}/results' / p.name, sha)
                    for gpu, p, sha in selected]
    for src, dst, sha in destinations:
        if dst.exists() and not os.path.samefile(src, dst):
            raise FileExistsError(dst)
        if src.stat().st_dev != dst.parent.stat().st_dev:
            raise ValueError('Hard links require the same filesystem')
    for src, dst, sha in destinations:
        if not dst.exists():
            os.link(src, dst)
        if not os.path.samefile(src, dst):
            raise ValueError(f'Link verification failed: {dst}')
    manifest = dict(source_directory=source.name, verified_files=records,
                    published_files=[dict(path=dst.relative_to(ROOT).as_posix(),
                                          source=src.relative_to(source).as_posix(),
                                          sha256=sha, bytes=src.stat().st_size)
                                     for src, dst, sha in destinations])
    (ROOT / 'RECOVERED_DATA.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Imported {len(destinations)} files using hard links; source data retained.')


if __name__ == '__main__':
    main()
