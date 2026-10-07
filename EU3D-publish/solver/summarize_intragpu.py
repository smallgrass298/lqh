#!/usr/bin/env python3
"""Summarise an intra-GPU ablation job.

Reads the per-mode logs produced by run_intragpu_common.sh and reports, per mode:
  * loop time statistics over the repeats
  * the paired change against the baseline mode
  * the DLB phase timers, traffic counters and the probe split when present

Deliberately does not invent numbers.  A field that a mode never printed is left
blank rather than filled with an estimate, and instrumented runs are flagged so
they cannot be quoted as performance results.
"""

import argparse
import glob
import os
import re
import statistics
import sys

LOOP_RE = re.compile(r"^GPU Loop time:\s*([0-9.eE+-]+)", re.M)
CONFIG_RE = re.compile(r"^GPU DLB config:\s*(.+)$", re.M)
TIMING_RE = re.compile(
    r"^GPU DLB timing\(max-rank cumulative s\):\s*plan=([0-9.eE+-]+)\s+"
    r"pack=([0-9.eE+-]+)\s+forward=([0-9.eE+-]+)\s+remote=([0-9.eE+-]+)\s+"
    r"backward=([0-9.eE+-]+)\s+unpack=([0-9.eE+-]+)", re.M)
SPLIT_RE = re.compile(
    r"^GPU DLB mpi split\(max-rank cumulative s\):\s*forward_skew=([0-9.eE+-]+)\s+"
    r"forward_transfer=([0-9.eE+-]+)\s+backward_skew=([0-9.eE+-]+)\s+"
    r"backward_transfer=([0-9.eE+-]+)", re.M)
COUNTERS_RE = re.compile(
    r"^GPU DLB counters:\s*replans=(\d+)\s+skipped=(\d+)\s+active_steps=(\d+)\s+"
    r"planned_tasks=(\d+)\s+gate_rejected=(\d+)\s+budget_stops=(\d+)", re.M)
SCHED_RE = re.compile(
    r"^GPU reaction sched counters:\s*bucket_evaluations=(\d+)\s+"
    r"bucket_bypasses=(\d+)", re.M)
TRAFFIC_RE = re.compile(
    r"^GPU DLB traffic\(global cumulative\):\s*forward_bytes=(\d+)\s+"
    r"backward_bytes=(\d+)\s+forward_messages=(\d+)\s+backward_messages=(\d+)", re.M)


def read_mode(result_dir, mode):
    logs = sorted(glob.glob(os.path.join(result_dir, f"gpu_{mode}_rep*.log")))
    info = {"mode": mode, "loops": [], "logs": len(logs)}
    for path in logs:
        with open(path, errors="replace") as handle:
            text = handle.read()
        loops = [float(v) for v in LOOP_RE.findall(text)]
        if loops:
            info["loops"].append(loops[-1])
        # Diagnostics come from the last repeat that printed them.
        m = CONFIG_RE.search(text)
        if m:
            info["config"] = m.group(1).strip()
        m = TIMING_RE.search(text)
        if m:
            info["timing"] = dict(zip(
                ("plan", "pack", "forward", "remote", "backward", "unpack"),
                (float(x) for x in m.groups())))
        m = SPLIT_RE.search(text)
        if m:
            info["split"] = dict(zip(
                ("forward_skew", "forward_transfer",
                 "backward_skew", "backward_transfer"),
                (float(x) for x in m.groups())))
        m = COUNTERS_RE.search(text)
        if m:
            info["counters"] = dict(zip(
                ("replans", "skipped", "active_steps", "planned_tasks",
                 "gate_rejected", "budget_stops"),
                (int(x) for x in m.groups())))
        m = SCHED_RE.search(text)
        if m:
            info["sched"] = dict(zip(
                ("bucket_evaluations", "bucket_bypasses"),
                (int(x) for x in m.groups())))
        m = TRAFFIC_RE.search(text)
        if m:
            info["traffic"] = dict(zip(
                ("forward_bytes", "backward_bytes",
                 "forward_messages", "backward_messages"),
                (int(x) for x in m.groups())))
    return info


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--baseline", default="off")
    args = ap.parse_args()

    modes = args.modes.split()
    data = {m: read_mode(args.result_dir, m) for m in modes}

    # Only the barrier-based skew timers indicate an instrumented run.  The
    # transfer timers are recorded on every path, so they must not be used to
    # detect the probe.
    probe = any(d.get("split", {}).get(k, 0.0) > 0.0
                for d in data.values()
                for k in ("forward_skew", "backward_skew"))

    print("=== EU3D-CUDA intra-GPU ablation ===")
    print(f"grid=281x141x32  steps={args.steps}  "
          f"gpus={args.gpus}  ranks={args.ranks}  mapping=one_gpu_per_rank")
    if probe:
        print("NOTE: EU3D_DLB_PROBE was active.  The MPI_Barrier instrumentation")
        print("      perturbs the runtime; use these loop times for mechanism only.")

    base = stats(data.get(args.baseline, {}).get("loops", []))

    print("\n-- loop time (s) --")
    header = (f"{'mode':<9}{'n':<4}{'mean':<12}{'median':<12}"
              f"{'stdev':<10}{'min':<12}{'max':<12}{'vs '+args.baseline:<26}")
    print(header)
    for m in modes:
        s = stats(data[m]["loops"])
        if not s:
            print(f"{m:<9}{'-':<4}{'no loop time parsed':<12}")
            continue
        delta = ""
        if base and m != args.baseline:
            saved = base["mean"] - s["mean"]
            pct = 100.0 * saved / base["mean"]
            word = "faster" if saved >= 0 else "slower"
            delta = f"{word} {abs(saved):.2f} s ({abs(pct):.2f}%)"
        print(f"{m:<9}{s['n']:<4}{s['mean']:<12.3f}{s['median']:<12.3f}"
              f"{s['stdev']:<10.3f}{s['min']:<12.3f}{s['max']:<12.3f}{delta:<26}")

    # Paired comparison against the frozen Stage 2 reference, which is the number
    # the supervisor actually asked about: does this branch beat Stage 2, not just
    # beat DLB-off.
    ref = stats(data.get("ref", {}).get("loops", []))
    if ref:
        print("\n-- vs frozen Stage 2 reference (mode 'ref') --")
        for m in modes:
            if m in ("off", "ref"):
                continue
            s = stats(data[m]["loops"])
            if not s:
                continue
            saved = ref["mean"] - s["mean"]
            pct = 100.0 * saved / ref["mean"]
            word = "faster" if saved >= 0 else "slower"
            print(f"{m:<9}{word} {abs(saved):.2f} s ({abs(pct):.2f}%) "
                  f"[ref mean {ref['mean']:.3f} -> {s['mean']:.3f}]")

    print("\n-- DLB phase timers (max-rank cumulative s; phases overlap, do not sum) --")
    keys = ("plan", "pack", "forward", "remote", "backward", "unpack")
    print(f"{'mode':<9}" + "".join(f"{k:<12}" for k in keys))
    for m in modes:
        t = data[m].get("timing")
        if not t:
            continue
        print(f"{m:<9}" + "".join(f"{t[k]:<12.3f}" for k in keys))

    if probe:
        print("\n-- MPI wait split (max-rank cumulative s) --")
        print("   skew = time spent waiting at the barrier for peers to arrive")
        print("   transfer = time spent in the exchange itself once all arrived")
        sk = ("forward_skew", "forward_transfer", "backward_skew", "backward_transfer")
        print(f"{'mode':<9}" + "".join(f"{k:<20}" for k in sk))
        for m in modes:
            s = data[m].get("split")
            if not s:
                continue
            print(f"{m:<9}" + "".join(f"{s[k]:<20.3f}" for k in sk))

    print("\n-- traffic --")
    print(f"{'mode':<9}{'fwd GiB':<11}{'bwd GiB':<11}{'fwd msgs':<11}"
          f"{'mean msg B':<13}{'eff MB/s':<11}")
    for m in modes:
        tr = data[m].get("traffic")
        if not tr:
            continue
        fwd = tr["forward_bytes"]
        msgs = tr["forward_messages"]
        mean_msg = fwd / msgs if msgs else 0.0
        t = data[m].get("timing")
        eff = fwd / t["forward"] / 1e6 if t and t["forward"] > 0 else 0.0
        print(f"{m:<9}{fwd/1024**3:<11.4f}{tr['backward_bytes']/1024**3:<11.4f}"
              f"{msgs:<11}{mean_msg:<13.1f}{eff:<11.3f}")

    print("\n-- planner / scheduler counters --")
    print(f"{'mode':<9}{'replans':<10}{'skipped':<10}{'active':<10}"
          f"{'tasks':<12}{'gate_rej':<12}{'budget':<9}{'buckets':<10}{'bypass':<8}")
    for m in modes:
        c = data[m].get("counters")
        if not c:
            continue
        sc = data[m].get("sched", {})
        print(f"{m:<9}{c['replans']:<10}{c['skipped']:<10}{c['active_steps']:<10}"
              f"{c['planned_tasks']:<12}{c['gate_rejected']:<12}"
              f"{c['budget_stops']:<9}"
              f"{sc.get('bucket_evaluations',''):<10}"
              f"{sc.get('bucket_bypasses',''):<8}")

    print("\n-- switch configuration reported by the binary --")
    for m in modes:
        if "config" in data[m]:
            print(f"{m:<9}{data[m]['config']}")

    exact = os.path.join(args.result_dir, "exact_match.txt")
    if os.path.exists(exact):
        print("\n-- output comparison --")
        with open(exact) as handle:
            sys.stdout.write(handle.read())

    print("\nReminder: quote clean (probe-off) runs for performance and keep the")
    print("instrumented runs for mechanism.  Phase timers are per-rank maxima and")
    print("overlap, so they must not be added into a wall-clock budget.")


if __name__ == "__main__":
    main()
