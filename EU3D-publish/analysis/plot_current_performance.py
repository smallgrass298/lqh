"""Plot the current dynamic modes from recorded full-run statistics."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    with (ROOT / 'results/summary/queue_performance.csv').open(newline='') as stream:
        rows = sorted((r for r in csv.DictReader(stream)
                       if r['mode'] in ('dynamic', 'dynamic_all')),
                      key=lambda r: int(r['gpus']))
    assert [int(r['gpus']) for r in rows] == [1, 2, 4, 8]
    assert all(r['complete'] == 'True' and int(r['n']) == 3 for r in rows)
    means = [float(r['mean_s']) for r in rows]
    deviations = [float(r['stdev_s']) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    bars = ax.bar(range(4), means, yerr=deviations, capsize=4, color='#309c70', width=.6)
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in means], padding=7)
    ax.set_xticks(range(4), ['1', '2', '4', '8'])
    ax.set_xlabel('GPUs (one MPI rank per GPU)')
    ax.set_ylabel('GPU loop time (s)')
    ax.set_ylim(0, max(means) * 1.18)
    ax.set_title('Dynamic warp queue\n281 × 141 × 32, 2834 steps, L40S; mean ± sample SD, n=3')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.2)
    output = ROOT / 'results/figures/current_performance.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
