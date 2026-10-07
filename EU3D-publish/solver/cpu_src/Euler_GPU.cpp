/* CPU initialization and output for the GPU solver. Arrays use row-major storage. */
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include "Euler.hpp"
#include "gpu_solver.h"

using namespace std;

void Euler::Computing_GPU(int count)
{
    int ni = Mymesh.ni, nj = Mymesh.nj, nk = Mymesh.nk, bc = Mymesh.bc;
    int NS = React.NS, NR = React.NR;

    if (cfl != 0.0)
    {
        if (myid == 0)
            cerr << "GPU solver currently supports the task-9.2 fixed-dt configuration only (cfl=0)." << endl;
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // Upload fields, reaction constants, grid coordinates and topology.
    // The GPU solver uses MPI_COMM_WORLD.
    gpu_solver_init(
        ni,nj,nk,bc,NS,NR,
        TwoDim.U.Getbuf(),TwoDim.V.Getbuf(),TwoDim.W.Getbuf(),TwoDim.P.Getbuf(),TwoDim.D.Getbuf(),
        TwoDim.T.Getbuf(),TwoDim.H.Getbuf(),TwoDim.E.Getbuf(),TwoDim.C.Getbuf(),TwoDim.Ma.Getbuf(),
        TwoDim.Gamma.Getbuf(),TwoDim.Cp.Getbuf(),TwoDim.Wav.Getbuf(),TwoDim.Rgas.Getbuf(),
        TwoDim.Yi.Getbuf(),TwoDim.Mc.Getbuf(),TwoDim.Di.Getbuf(),TwoDim.CS.Getbuf(),
        React.Stoi_F.Getbuf(),React.Stoi_B.Getbuf(),React.React_TB.Getbuf(),
        React.Af.Getbuf(),React.Bf.Getbuf(),React.Eaf.Getbuf(),
        React.Mw.Getbuf(),React.Ri.Getbuf(),React.Coeff0.Getbuf(),React.Coeff1.Getbuf(),
        Mymesh.xnode.Getbuf(),Mymesh.ynode.Getbuf(),Mymesh.znode.Getbuf(),
        8.31434, 1.987, 101325.0,
        myid_x,myid_y,myid_z,m_block_x,m_block_y,m_block_z,
        m_left,m_right,m_front,m_back,m_down,m_up,DLB_step,DLB_tol);

    double time = 0.0;
    int num = 0;
    iteration = 0;
    const long long fixed_steps =
        static_cast<long long>(std::ceil(Final_Time / dt - 1.0e-12));
    const bool dlb_disabled = [] {
        const char* value = std::getenv("EU3D_DISABLE_DLB");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool monitor_dlb = !dlb_disabled && numprocs > 1 && DLB_step > 0;
    double tt1 = MPI_Wtime();

    // Record load imbalance.
    ofstream loadfile;
    if (myid == 0 && monitor_dlb)
        loadfile.open("./output_" + to_string(numprocs) + "/load/LoadDegree.dat");

    while (iteration < fixed_steps)
    {
        double step_dt = (cfl == 0) ? dt : TwoDim.dt;

        // Download fields for CPU output.
        if (time <= num * Final_Time / count && time + step_dt > num * Final_Time / count)
        {
            gpu_solver_download(
                TwoDim.U.Getbuf(),TwoDim.V.Getbuf(),TwoDim.W.Getbuf(),TwoDim.P.Getbuf(),
                TwoDim.D.Getbuf(),TwoDim.T.Getbuf(),TwoDim.Yi.Getbuf(),TwoDim.Gamma.Getbuf(),
                TwoDim.Ma.Getbuf());
            Output(num);
            num++;
        }

        // Advance boundary, exchange, advection and reaction stages.
        gpu_solver_step(step_dt, iteration);

        iteration++;

        // Sample rank load every DLB_step iterations.
        if (monitor_dlb && iteration % DLB_step == 0)
        {
            long long localNchem = gpu_solver_sum_nchem();
            int localNchemMax = gpu_solver_max_nchem();
            int globalNchemMax = 0;
            MPI_Allreduce(&localNchemMax, &globalNchemMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            int localNchemInt = (int)localNchem;
            MPI_Allgather(&localNchemInt, 1, MPI_INT, &React.NchemTotal(0), 1, MPI_INT, MPI_COMM_WORLD);
            DLB.LoadDegree = double(React.NchemTotal.MaxValue() - React.NchemTotal.AveValue())
                             / React.NchemTotal.MaxValue();
            if (myid == 0)
                cout << "Load Degree: " << DLB.LoadDegree
                     << "  NchemMax: " << globalNchemMax << endl;
            if (myid == 0)
                loadfile << iteration << '\t' << DLB.LoadDegree << '\t' << globalNchemMax << '\n';
        }

        time += step_dt;
        if (myid == 0)
            cout << "GPU Iteration: " << iteration << "  dt=" << step_dt << "  t=" << time << endl;
    }

    // The optimized single-rank path queues consecutive kernels asynchronously.
    // Synchronize once at the timing boundary so the measured loop is complete.
    gpu_solver_synchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    trackt = MPI_Wtime() - tt1;
    if (myid == 0)
        cout << "GPU Loop time: " << trackt << endl;

    // Download final fields; the caller writes Output(count).
    gpu_solver_download(
        TwoDim.U.Getbuf(),TwoDim.V.Getbuf(),TwoDim.W.Getbuf(),TwoDim.P.Getbuf(),
        TwoDim.D.Getbuf(),TwoDim.T.Getbuf(),TwoDim.Yi.Getbuf(),TwoDim.Gamma.Getbuf(),
        TwoDim.Ma.Getbuf());
    gpu_solver_finalize();
}
