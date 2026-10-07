#!/usr/bin/env python3
"""Standard-library reports for completed or partial queue suite jobs.
Usage: python3 summarize_queue_suite.py [intragpu_results or one full result dir]
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path
from summarize_intragpu import LOOP_RE, TIMING_RE, TRAFFIC_RE, COUNTERS_RE


def save(path, rows, fields):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def report(root):
    dirs=[root] if root.name.endswith('_suite_full') else sorted(root.glob('*_suite_full'))
    runs=[]; sched=[]; comparisons=[]; statuses=[]
    for d in dirs:
        meta=dict(line.split('=',1) for line in (d/'resource.log').read_text().splitlines() if '=' in line)
        gpu=int(meta['physical_gpus']); expected=int(meta['repeats'])
        modes=meta['modes'].split(); by_mode={m:{} for m in modes}
        finished=(d/'correctness.txt').exists() and (d/'summary.txt').exists()
        for mode in modes:
            for log in sorted(d.glob(f'gpu_{mode}_rep*.log')):
                rep=int(re.search(r'_rep(\d+)\.log$',log.name)[1]); text=log.read_text()
                loops=LOOP_RE.findall(text)
                if not loops: continue
                value=float(loops[-1]); by_mode[mode][rep]=value
                row=dict(job=d.name,gpus=gpu,mode=mode,repeat=rep,loop_s=value)
                for regex, keys in [
                    (TIMING_RE,('plan_s','pack_s','forward_s','remote_s','backward_s','unpack_s')),
                    (TRAFFIC_RE,('forward_bytes','backward_bytes','forward_messages','backward_messages')),
                    (COUNTERS_RE,('replans','skipped','active_steps','planned_tasks','gate_rejected','budget_stops'))]:
                    match=regex.search(text)
                    if match: row.update(zip(keys, match.groups()))
                runs.append(row)
                pattern=r'GPU reaction scheduler rank=(\d+) mode=(\w+) dynamic_launches=(\d+) bucket_evaluations=(\d+) bucket_bypasses=(\d+) worker_blocks=(\d+) chunk=(\d+)'
                entries=list(re.finditer(pattern,text))
                if len({m[1] for m in entries})!=gpu: finished=False
                for m in entries:
                    item=dict(job=d.name,gpus=gpu,mode=mode,repeat=rep)
                    item.update(zip(('rank','scheduler','dynamic_launches','bucket_evaluations','bucket_bypasses','worker_blocks','chunk'),m.groups()))
                    sched.append(item)
        complete=finished and all(set(v)==set(range(1,expected+1)) for v in by_mode.values())
        statuses.append(f'{d.name}: {"COMPLETE" if complete else "INCOMPLETE — inspect logs"}')
        for mode in modes:
            values=list(by_mode[mode].values())
            if not values: continue
            paired=sorted(set(by_mode[modes[0]]) & set(by_mode[mode]))
            base=[by_mode[modes[0]][i] for i in paired]
            new=[by_mode[mode][i] for i in paired]
            comparisons.append(dict(job=d.name,gpus=gpu,mode=mode,complete=complete,n=len(values),
                mean_s=st.mean(values),median_s=st.median(values),stdev_s=st.stdev(values) if len(values)>1 else 0,
                min_s=min(values),max_s=max(values),baseline=modes[0],paired_n=len(paired),
                time_reduction_pct=100*(st.mean(base)-st.mean(new))/st.mean(base) if paired else '',
                speedup=st.mean(base)/st.mean(new) if paired else '',
                paired_wins=sum(n<b for b,n in zip(base,new)),paired_losses=sum(n>b for b,n in zip(base,new))))
    root.mkdir(exist_ok=True)
    save(root/'queue_runs.csv',runs,['job','gpus','mode','repeat','loop_s','plan_s','pack_s','forward_s','remote_s','backward_s','unpack_s','forward_bytes','backward_bytes','forward_messages','backward_messages','replans','skipped','active_steps','planned_tasks','gate_rejected','budget_stops'])
    save(root/'queue_scheduler.csv',sched,['job','gpus','mode','repeat','rank','scheduler','dynamic_launches','bucket_evaluations','bucket_bypasses','worker_blocks','chunk'])
    save(root/'queue_performance.csv',comparisons,['job','gpus','mode','complete','n','mean_s','median_s','stdev_s','min_s','max_s','baseline','paired_n','time_reduction_pct','speedup','paired_wins','paired_losses'])
    lines=['Warp queue suite — 281x141x32 / 2834 steps / 1 GPU per rank',*statuses,'',
           'GPU mode             n mean(s)   stdev(s) reduction vs baseline  paired wins']
    for c in comparisons:
        reduction=c['time_reduction_pct']
        lines.append(f"{c['gpus']:3} {c['mode']:16} {c['n']} {c['mean_s']:9.3f} {c['stdev_s']:8.3f} {str(round(reduction,3))+'%' if reduction!='' else 'N/A':>12}  {c['paired_wins']}/{c['paired_n']}")
    if not dirs: lines.append('No full-suite result directories found yet.')
    lines += ['', 'INCOMPLETE rows are provisional; use only COMPLETE jobs for conclusions.',
              'Positive reduction means faster. Baseline: sorted (1 GPU), all (2/4/8 GPU).',
              'Phase timers are cumulative per-rank maxima and overlap: do not sum them.',
              'Scheduler counts are per rank; zero dynamic_launches means the path bypassed.',
              'No hardware occupancy, warp efficiency or SM utilisation was measured.']
    output='\n'.join(lines)+'\n'; (root/'queue_report.txt').write_text(output); print(output)

if __name__=='__main__': report(Path(sys.argv[1] if len(sys.argv)>1 else 'intragpu_results'))
