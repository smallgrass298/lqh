#!/bin/bash
# ==============================================================================
# Intra-GPU / communication-overlap ablation runner.
#
# Environment switches select each mode from one build.
#
# Modes
#   off      DLB disabled                        (speedup denominator)
#   ref      frozen Stage 2 behaviour            (correctness + performance reference)
#   overlap  ref + EU3D_DLB_SCHEDULE=overlap     (track c: real compute/comm overlap)
#   gate     ref + EU3D_DLB_COST_GATE=1          (track c: stop unprofitable migrations)
#   comm     ref + overlap + gate                (all of track c)
#   sorted   ref + EU3D_REACTION_SCHED=sorted    (track d: warp-uniform local reaction)
#   taskord  ref + EU3D_DLB_TASK_ORDER=nchem     (track d: warp-uniform migrated solve)
#   warp     ref + sorted + taskord              (all of track d)
#   all      ref + overlap + gate + sorted + taskord
#
# Required environment (set by the PBS wrapper):
#   EU3D_GPUS EU3D_RANKS EU3D_STEPS EU3D_REPEATS EU3D_CTRL_FILE EU3D_TAG EU3D_MODES
# Optional:
#   EU3D_PROBE=1   add MPI_Barrier instrumentation to split skew from transfer.
#                  Diagnostic only: the barriers perturb the runtime, so a probe
#                  run must never be quoted as a performance result.
# ==============================================================================
set -euo pipefail

: "${PBS_O_WORKDIR:?submit through PBS}"
: "${EU3D_GPUS:?wrapper must set EU3D_GPUS}"
: "${EU3D_RANKS:?wrapper must set EU3D_RANKS}"
: "${EU3D_STEPS:?wrapper must set EU3D_STEPS}"
: "${EU3D_REPEATS:?wrapper must set EU3D_REPEATS}"
: "${EU3D_CTRL_FILE:?wrapper must set EU3D_CTRL_FILE}"
: "${EU3D_TAG:?wrapper must set EU3D_TAG}"
: "${EU3D_MODES:?wrapper must set EU3D_MODES}"
EU3D_PROBE="${EU3D_PROBE:-0}"

if [[ "$EU3D_GPUS" != "$EU3D_RANKS" ]]; then
    echo "ERROR: this branch keeps the one-physical-GPU-per-MPI-rank mapping" >&2
    exit 2
fi

module purge
module load tools/prod
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
export OMPI_CC=gcc
export OMPI_CXX=g++
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n
export EU3D_CTRL="./input/${EU3D_CTRL_FILE}"

job_id="${PBS_JOBID:-manual}"
result_dir="$PBS_O_WORKDIR/intragpu_results/${job_id}_${EU3D_TAG}"
work_dir="${TMPDIR:-/tmp}/eu3d_intragpu_${job_id}_${EU3D_TAG}"
mkdir -p "$result_dir" "$work_dir"

cp -R "$PBS_O_WORKDIR/cpu_src" "$PBS_O_WORKDIR/include" \
      "$PBS_O_WORKDIR/input" "$PBS_O_WORKDIR/src" "$work_dir/"
cp "$PBS_O_WORKDIR/Makefile" "$PBS_O_WORKDIR/compare_results.py" \
   "$PBS_O_WORKDIR/summarize_intragpu.py" "$PBS_O_WORKDIR/verify_queue_outputs.py" "$work_dir/"
cd "$work_dir"

visible="$(nvidia-smi -L | wc -l | tr -d ' ')"
{
    echo "job=$job_id"
    echo "grid=281x141x32"
    echo "steps=$EU3D_STEPS"
    echo "mpi_ranks=$EU3D_RANKS"
    echo "physical_gpus=$EU3D_GPUS"
    echo "mapping=one_gpu_per_rank"
    echo "ctrl=$EU3D_CTRL"
    echo "modes=$EU3D_MODES"
    echo "repeats=$EU3D_REPEATS"
    echo "probe=$EU3D_PROBE"
    echo "host=$(hostname)"
    echo "visible_gpus=$visible"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
} | tee "$result_dir/resource.log"

[[ "$visible" == "$EU3D_GPUS" ]] || {
    echo "ERROR: requested $EU3D_GPUS GPUs but CUDA sees $visible" \
        | tee -a "$result_dir/resource.log"
    exit 8
}

make -j "$EU3D_RANKS" ARCH=-arch=sm_89 NUMERIC_MODE=native 2>&1 \
    | tee "$result_dir/build.log"

# Validate the host planner before running the solver.
make test_dlb_schedule 2>&1 | tee -a "$result_dir/build.log"
./test_dlb_schedule 2>&1 | tee "$result_dir/test_dlb_schedule.log"

make test_warp_queue 2>&1 | tee -a "$result_dir/build.log"
./test_warp_queue 2>&1 | tee "$result_dir/test_warp_queue.log"
if [[ "${EU3D_QUEUE_SANITIZE:-0}" == "1" ]]; then
    compute-sanitizer --tool memcheck --error-exitcode 12 ./test_warp_queue
    compute-sanitizer --tool synccheck --error-exitcode 12 ./test_warp_queue
fi

clear_switches() {
    unset EU3D_DISABLE_DLB      || true
    unset EU3D_CUDA_AWARE_MPI   || true
    unset EU3D_DLB_SCHEDULE     || true
    unset EU3D_DLB_TASK_ORDER   || true
    unset EU3D_REACTION_SCHED   || true
    unset EU3D_DLB_COST_GATE    || true
    unset EU3D_REACTION_SCHED_MIN_HEAVY || true
    unset EU3D_DLB_PROBE        || true
}

apply_mode() {
    local mode="$1"
    clear_switches
    # Every DLB-on mode uses the CUDA-aware transport, which is the Stage 2
    # recommended baseline.
    case "$mode" in
        dynamic) export EU3D_CUDA_AWARE_MPI=1 EU3D_REACTION_SCHED=dynamic ;;
        sorted_force) export EU3D_CUDA_AWARE_MPI=1 EU3D_REACTION_SCHED=sorted EU3D_REACTION_SCHED_MIN_HEAVY=0 ;;
        dynamic_force) export EU3D_CUDA_AWARE_MPI=1 EU3D_REACTION_SCHED=dynamic EU3D_REACTION_SCHED_MIN_HEAVY=0 ;;
        dynamic_all) export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_SCHEDULE=overlap \
                            EU3D_DLB_COST_GATE=1 EU3D_REACTION_SCHED=dynamic \
                            EU3D_DLB_TASK_ORDER=nchem ;;
        off)      export EU3D_DISABLE_DLB=1 ;;
        ref)      export EU3D_CUDA_AWARE_MPI=1 ;;
        overlap)  export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_SCHEDULE=overlap ;;
        gate)     export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_COST_GATE=1 ;;
        comm)     export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_SCHEDULE=overlap \
                         EU3D_DLB_COST_GATE=1 ;;
        sorted)   export EU3D_CUDA_AWARE_MPI=1 EU3D_REACTION_SCHED=sorted ;;
        taskord)  export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_TASK_ORDER=nchem ;;
        warp)     export EU3D_CUDA_AWARE_MPI=1 EU3D_REACTION_SCHED=sorted \
                         EU3D_DLB_TASK_ORDER=nchem ;;
        all)      export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_SCHEDULE=overlap \
                         EU3D_DLB_COST_GATE=1 EU3D_REACTION_SCHED=sorted \
                         EU3D_DLB_TASK_ORDER=nchem ;;
        # Probe variants live in the same job so the barrier split does not need a
        # second queue slot.  Their loop times are instrumented and must only be
        # read for mechanism.
        refprobe) export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_PROBE=1 ;;
        allprobe) export EU3D_CUDA_AWARE_MPI=1 EU3D_DLB_SCHEDULE=overlap \
                         EU3D_DLB_COST_GATE=1 EU3D_REACTION_SCHED=sorted \
                         EU3D_DLB_TASK_ORDER=nchem EU3D_DLB_PROBE=1 ;;
        *) echo "ERROR: unknown mode $mode" >&2; exit 3 ;;
    esac
    [[ "$EU3D_PROBE" == "1" ]] && export EU3D_DLB_PROBE=1
    return 0
}

run_mode() {
    local mode="$1"
    apply_mode "$mode"
    local rep="$2"
        local log="$result_dir/gpu_${mode}_rep${rep}.log"
        {
            echo "===== mode=$mode repeat=$rep ====="
            echo "switches: DISABLE_DLB=${EU3D_DISABLE_DLB:-} CUDA_AWARE=${EU3D_CUDA_AWARE_MPI:-}"
            echo "          SCHEDULE=${EU3D_DLB_SCHEDULE:-} TASK_ORDER=${EU3D_DLB_TASK_ORDER:-}"
            echo "          REACTION_SCHED=${EU3D_REACTION_SCHED:-} COST_GATE=${EU3D_DLB_COST_GATE:-}"
            echo "          PROBE=${EU3D_DLB_PROBE:-}"
        } | tee "$log"
        rm -f "output_${EU3D_RANKS}"/results/*.dat
        mpirun -n "$EU3D_RANKS" --bind-to core ./EU3D_GPU 2>&1 | tee -a "$log"
        python3 - "$log" "$EU3D_STEPS" "$mode" <<'CHECKLOG'
import re, sys
text = open(sys.argv[1]).read()
steps = re.findall(r'^GPU Iteration: (\d+)', text, re.M)
if not steps or int(steps[-1]) != int(sys.argv[2]):
    sys.exit('FAIL: run did not complete the requested step count')
if not re.search(r'^GPU Loop time:', text, re.M):
    sys.exit('FAIL: missing completed GPU timing')
if sys.argv[3] == 'dynamic_force':
    counts = re.findall(r'dynamic_launches=(\d+)', text)
    if not counts or max(map(int, counts)) == 0:
        sys.exit('FAIL: forced dynamic mode never launched')
CHECKLOG
        snapshot="$result_dir/results_${mode}_rep${rep}"
        mkdir -p "$snapshot"
        cp "output_${EU3D_RANKS}"/results/*.dat "$snapshot/"
        reference="$result_dir/results_${first_mode}_rep1"
        python3 verify_queue_outputs.py "$reference" "$snapshot" "$EU3D_RANKS" \
            | tee "$result_dir/check_${mode}_rep${rep}.txt"
    mkdir -p "$result_dir/results_${mode}"
    cp "output_${EU3D_RANKS}"/results/*.dat "$result_dir/results_${mode}/"
}

first_mode="${EU3D_MODES%% *}"
read -r -a modes <<< "$EU3D_MODES"
for rep in $(seq 1 "$EU3D_REPEATS"); do
    if (( rep % 2 )); then
        for mode in "${modes[@]}"; do run_mode "$mode" "$rep"; done
    else
        for ((i=${#modes[@]}-1;i>=0;i--)); do run_mode "${modes[$i]}" "$rep"; done
    fi
done

echo "PASS: every repeat has complete finite outputs byte-identical to $first_mode rep1" | tee "$result_dir/correctness.txt"
echo "all_modes_byte_identical=yes" > "$result_dir/exact_match.txt"

python3 summarize_intragpu.py \
    --result-dir "$result_dir" \
    --gpus "$EU3D_GPUS" --ranks "$EU3D_RANKS" --steps "$EU3D_STEPS" \
    --modes "$EU3D_MODES" --baseline "$first_mode" \
    | tee "$result_dir/summary.txt"

echo "DONE: $result_dir"
