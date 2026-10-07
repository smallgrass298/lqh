#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
record="queue_submissions_$(date +%Y%m%d_%H%M%S).txt"
# Independent jobs: no prerequisite allocation or second round of queueing.
for gpu in 1 2 4 8; do
    job=$(qsub "run_queue_suite_${gpu}gpu.pbs")
    printf '%s GPU: %s\n' "$gpu" "$job" | tee -a "$record"
done
echo "Submission record: $record"
