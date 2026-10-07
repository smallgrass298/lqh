#include "dlb_schedule.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <set>

int main() {
    const int ranks=8,cells=16;
    std::vector<int> counts(ranks,cells),displs(ranks);
    for (int r=1;r<ranks;r++) displs[r]=displs[r-1]+counts[r-1];
    std::vector<int> weights(ranks*cells,1);
    weights[0*cells+3]=80;
    weights[0*cells+7]=40;
    weights[1*cells+2]=25;
    auto result=build_dlb_schedule(weights,counts,displs,0.001);
    assert(!result.plan.empty());
    auto before_range=*std::max_element(result.before.begin(),result.before.end())
                     -*std::min_element(result.before.begin(),result.before.end());
    auto after_range=*std::max_element(result.after.begin(),result.after.end())
                    -*std::min_element(result.after.begin(),result.after.end());
    assert(after_range<before_range);
    assert(std::accumulate(result.before.begin(),result.before.end(),0LL)==
           std::accumulate(result.after.begin(),result.after.end(),0LL));
    std::set<std::pair<int,int>> sent;
    for (size_t p=0;p<result.plan.size();p+=3) {
        assert(result.plan[p]!=result.plan[p+1]);
        assert(sent.insert({result.plan[p],result.plan[p+2]}).second);
    }

    std::vector<int> balanced(ranks*cells,3);
    auto no_work=build_dlb_schedule(balanced,counts,displs,0.001);
    assert(no_work.plan.empty());
    assert(no_work.before==no_work.after);

    // Representative 281x141x32 two-rank planner workload.  This guards the
    // bucketed implementation against accidentally returning to a per-task
    // full-grid scan while also checking conservation and balance improvement.
    const int large_cells=(281*141*32)/2;
    std::vector<int> large_counts(2,large_cells),large_displs{0,large_cells};
    std::vector<int> large_weights(2*large_cells,4);
    for (int q=0;q<large_cells;q+=97) large_weights[q]=80+(q%7);
    auto start=std::chrono::steady_clock::now();
    auto large=build_dlb_schedule(large_weights,large_counts,large_displs,0.01);
    double elapsed=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    auto large_before_range=*std::max_element(large.before.begin(),large.before.end())
                           -*std::min_element(large.before.begin(),large.before.end());
    auto large_after_range=*std::max_element(large.after.begin(),large.after.end())
                          -*std::min_element(large.after.begin(),large.after.end());
    assert(!large.plan.empty());
    assert(large_after_range<large_before_range);
    assert(std::accumulate(large.before.begin(),large.before.end(),0LL)==
           std::accumulate(large.after.begin(),large.after.end(),0LL));

    // ---- Cost gate: default config must reproduce the ungated planner ----
    {
        DlbGateConfig defaults;
        defaults.tolerance=0.001;
        auto gated=build_dlb_schedule(weights,counts,displs,defaults);
        assert(gated.plan==result.plan);
        assert(gated.before==result.before);
        assert(gated.after==result.after);
        assert(gated.rejected_by_weight_gate==0);
        assert(!gated.stopped_on_budget);
    }

    // ---- Cost gate: min_task_weight must never migrate a light task ----
    // Mirrors the measured 8-GPU situation: a huge population of Nchem=1 cells
    // plus a small number of genuinely heavy cells.  Without a gate the planner
    // happily ships Nchem=1 cells across MPI to shave the last few percent of
    // LoadDegree; with the gate those cells must stay local.
    {
        const int r2=4,c2=64;
        std::vector<int> c(r2,c2),dsp(r2,0);
        for (int r=1;r<r2;r++) dsp[r]=dsp[r-1]+c[r-1];
        std::vector<int> w(r2*c2,1);
        for (int q=0;q<12;q++) w[0*c2+q]=60+q;   // heavy cells concentrated on rank 0
        for (int q=0;q<6;q++)  w[1*c2+q]=15+q;   // medium cells on rank 1

        DlbGateConfig open_gate; open_gate.tolerance=0.001;
        auto ungated=build_dlb_schedule(w,c,dsp,open_gate);

        DlbGateConfig gate; gate.tolerance=0.001; gate.min_task_weight=10;
        auto gated=build_dlb_schedule(w,c,dsp,gate);

        // Load accounting must still include the light cells.
        assert(gated.before==ungated.before);
        assert(std::accumulate(gated.before.begin(),gated.before.end(),0LL)==
               std::accumulate(gated.after.begin(),gated.after.end(),0LL));
        // Every migrated task must clear the weight gate.
        for (size_t p=0;p<gated.plan.size();p+=3) {
            int sender=gated.plan[p], cell=gated.plan[p+2];
            assert(w[dsp[sender]+cell]>=10);
        }
        // The gate is what makes the plan smaller; that is the whole point.
        assert(gated.plan.size()<=ungated.plan.size());
        assert(gated.rejected_by_weight_gate>0);
        std::cout << "weight gate: tasks " << ungated.plan.size()/3 << " -> "
                  << gated.plan.size()/3 << " (rejected "
                  << gated.rejected_by_weight_gate << " light candidates)\n";
    }

    // ---- Cost gate: task budget must cap the plan and be reported ----
    {
        DlbGateConfig gate;
        gate.tolerance=0.001;
        gate.max_tasks_per_replan=2;
        auto capped=build_dlb_schedule(weights,counts,displs,gate);
        assert(capped.plan.size()/3<=2);
        assert(std::accumulate(capped.before.begin(),capped.before.end(),0LL)==
               std::accumulate(capped.after.begin(),capped.after.end(),0LL));
        std::cout << "task budget: plan capped at " << capped.plan.size()/3
                  << " tasks, stopped_on_budget=" << (capped.stopped_on_budget?1:0)
                  << '\n';
    }

    // ---- Cost gate: a rank made entirely of light cells offers nothing ----
    {
        const int r3=2,c3=32;
        std::vector<int> c(r3,c3),dsp(r3,0);
        dsp[1]=c3;
        std::vector<int> w(r3*c3,1);
        for (int q=0;q<c3;q++) w[q]=2;   // rank 0 is heavier but still all-light
        DlbGateConfig gate; gate.tolerance=0.001; gate.min_task_weight=10;
        auto none=build_dlb_schedule(w,c,dsp,gate);
        assert(none.plan.empty());
        assert(none.before==none.after);
    }

    std::cout << "DLB scheduler OK: tasks=" << result.plan.size()/3
              << " load-range=" << before_range << " -> " << after_range << '\n'
              << "large-grid planner: cells=" << large_weights.size()
              << " tasks=" << large.plan.size()/3
              << " load-range=" << large_before_range << " -> " << large_after_range
              << " time=" << elapsed << " s\n";
}
