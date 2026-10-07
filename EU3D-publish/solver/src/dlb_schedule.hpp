#ifndef EU3D_DLB_SCHEDULE_HPP
#define EU3D_DLB_SCHEDULE_HPP

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

struct DlbScheduleResult {
    // Repeated triples: sender rank, receiver rank, sender-local compact cell.
    std::vector<int> plan;
    std::vector<long long> before;
    std::vector<long long> after;
    // Gate bookkeeping so callers can report why the planner stopped.
    long long rejected_by_weight_gate = 0;
    long long rejected_by_budget = 0;
    bool stopped_on_budget = false;
};

/* Cost-awareness configuration for the cross-rank planner.
 *
 * Background from the instrumented 8-GPU run (281x141x32, 2834 steps):
 * the planner moved about 3413 tasks per replan, but only about 872 interior
 * cells per step actually had Nchem >= 10.  The remaining migrated tasks were
 * Nchem 1-9 cells, i.e. moving them buys a handful of reaction sub-iterations
 * while still paying a full 320-byte round trip plus its share of the
 * synchronisation cost.  min_task_weight and max_tasks_per_replan exist to stop
 * the planner from buying those unprofitable trades purely to drive LoadDegree
 * towards zero.
 *
 * Defaults reproduce the frozen Stage 2 planner exactly:
 *   min_task_weight = 1        -> every positive weight is eligible
 *   max_tasks_per_replan = 0   -> no budget
 */
struct DlbGateConfig {
    double tolerance = 0.001;
    int min_task_weight = 1;
    long long max_tasks_per_replan = 0;
};

inline DlbScheduleResult build_dlb_schedule(const std::vector<int>& all_nchem,
                                            const std::vector<int>& counts,
                                            const std::vector<int>& displs,
                                            const DlbGateConfig& gate) {
    const int nranks=(int)counts.size();
    DlbScheduleResult result;
    result.before.assign(nranks,0);
    const int min_weight=gate.min_task_weight>1?gate.min_task_weight:1;
    // Nchem is an integer iteration count with far fewer distinct values than
    // mesh cells.  Keep available cells in ordered weight buckets instead of
    // sorting every cell and rescanning the full rank for each migrated task.
    // This changes planner complexity from repeated O(cells) scans to O(log W)
    // lookup, where W is the number of distinct Nchem values.
    //
    // The weight gate is applied when filling the buckets rather than inside
    // the selection loop: cells below min_task_weight still count towards the
    // rank's load (they are real work) but they are never offered as migration
    // candidates.
    std::vector<std::map<int,std::vector<int>>> available(nranks);
    for (int r=0;r<nranks;r++) {
        for (int q=0;q<counts[r];q++) {
            int weight=all_nchem[displs[r]+q];
            result.before[r]+=weight;
            if (weight<=0) continue;
            if (weight<min_weight) { result.rejected_by_weight_gate++; continue; }
            available[r][weight].push_back(q);
        }
    }
    result.after=result.before;
    const size_t hard_limit=(size_t)std::accumulate(counts.begin(),counts.end(),0);
    const size_t budget=gate.max_tasks_per_replan>0
        ? std::min<size_t>(hard_limit,(size_t)gate.max_tasks_per_replan)
        : hard_limit;
    while (result.plan.size()/3<budget) {
        int sender=(int)std::distance(result.after.begin(),
                         std::max_element(result.after.begin(),result.after.end()));
        int receiver=(int)std::distance(result.after.begin(),
                           std::min_element(result.after.begin(),result.after.end()));
        long long high=result.after[sender],low=result.after[receiver],diff=high-low;
        if (high<=0 || (double)diff/(double)high<=gate.tolerance) break;

        auto& buckets=available[sender];
        if (buckets.empty()) break;

        // Moving a task of weight w changes the sender/receiver difference
        // from diff to |diff-2w|.  Therefore only the bucket(s) immediately
        // around diff/2 can be optimal.
        const long long target_ll=(diff+1)/2;
        const int target=(int)std::min<long long>(
            target_ll,std::numeric_limits<int>::max());
        auto upper=buckets.lower_bound(target);
        auto best_it=buckets.end();
        long long best_pair_diff=diff;
        int best_weight=0;
        auto consider=[&](auto it) {
            if (it==buckets.end()) return;
            int weight=it->first;
            if (weight<=0 || (long long)weight>=diff) return;
            long long pair_diff=std::llabs(diff-2LL*weight);
            if (pair_diff<best_pair_diff ||
                (pair_diff==best_pair_diff && weight>best_weight)) {
                best_pair_diff=pair_diff;
                best_weight=weight;
                best_it=it;
            }
        };
        consider(upper);
        if (upper!=buckets.begin()) consider(std::prev(upper));
        if (best_it==buckets.end()) break;

        int best_cell=best_it->second.back();
        best_it->second.pop_back();
        if (best_it->second.empty()) buckets.erase(best_it);
        result.after[sender]-=best_weight;
        result.after[receiver]+=best_weight;
        result.plan.push_back(sender);
        result.plan.push_back(receiver);
        result.plan.push_back(best_cell);
    }
    if (budget<hard_limit && result.plan.size()/3>=budget) {
        // Report that balancing stopped because of the migration budget rather
        // than because the tolerance was met.
        int sender=(int)std::distance(result.after.begin(),
                         std::max_element(result.after.begin(),result.after.end()));
        int receiver=(int)std::distance(result.after.begin(),
                           std::min_element(result.after.begin(),result.after.end()));
        long long high=result.after[sender],low=result.after[receiver];
        if (high>0 && (double)(high-low)/(double)high>gate.tolerance)
            result.stopped_on_budget=true;
    }
    return result;
}

// Backwards-compatible entry point used by the frozen Stage 2 call sites and by
// test_dlb_schedule.  Behaviour is bit-for-bit the ungated planner.
inline DlbScheduleResult build_dlb_schedule(const std::vector<int>& all_nchem,
                                            const std::vector<int>& counts,
                                            const std::vector<int>& displs,
                                            double tolerance) {
    DlbGateConfig gate;
    gate.tolerance=tolerance;
    return build_dlb_schedule(all_nchem,counts,displs,gate);
}

#endif
