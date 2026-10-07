/* Device memory management and time stepping. CPU code handles initialization and output; fields remain on the GPU between downloads. Storage matches CPU row-major arrays. */
#include "kernels.cuh"
#include "thermo.cuh"   // THERMO_NCOEF
#include <cstdlib>
#include <cstring>

namespace {

Dims      D;
DevFields F;
DevConst  C;
Topo      T;
double*   g_dsend = nullptr;
double*   g_drecv = nullptr;
MPI_Comm  g_comm;
bool      g_init = false;
bool      g_defer_step_sync = false;
int       g_dlb_step = 5;
double    g_dlb_tol = 0.001;

size_t N3() { return (size_t)D.Ni*D.Nj*D.Nk; }
size_t N4() { return N3()*D.NC; }
size_t NSsz(){ return N3()*D.NS; }

double* dmalloc(size_t n) { double* p; CUDA_CHECK(cudaMalloc(&p, n*sizeof(double))); return p; }
void up(double* dst, const double* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n*sizeof(double), cudaMemcpyHostToDevice));
}
void down(double* src, double* dst, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n*sizeof(double), cudaMemcpyDeviceToHost));
}

} // namespace

/* Initialize device fields from host Array::Getbuf() pointers. */
extern "C" void gpu_solver_init(
    int ni,int nj,int nk,int bc,int NS,int NR,
    // 3D fields.
    const double* U,const double* V,const double* W,const double* P,const double* Dd,
    const double* Tt,const double* H,const double* E,const double* Cc,const double* Ma,
    const double* Gamma,const double* Cp,const double* Wav,const double* Rgas,
    // 4D fields.
    const double* Yi,const double* Mc,const double* Di,const double* CS,
    // Reaction constants.
    const double* Stoi_F,const double* Stoi_B,const double* React_TB,
    const double* Af,const double* Bf,const double* Eaf,
    const double* Mw,const double* Ri,const double* Coeff0,const double* Coeff1,
    const double* xnode,const double* ynode,const double* znode,
    double Rconst,double Ru,double P0,
    // Rank topology.
    int myid_x,int myid_y,int myid_z,int mbx,int mby,int mbz,
    int m_left,int m_right,int m_front,int m_back,int m_down,int m_up,
    int dlb_step,double dlb_tol)
{
    if (NS > 16 || NR > 24) {
        fprintf(stderr, "Unsupported chemistry size: NS=%d NR=%d (limits: NS<=16, NR<=24)\n", NS, NR);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    // Map node-local ranks cyclically to visible GPUs.
    // Multiple ranks may share a GPU.
    {
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "No visible CUDA device for MPI rank\n");
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
        int local_rank = 0;
        const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");   // OpenMPI
        if (!lr) lr = getenv("MV2_COMM_WORLD_LOCAL_RANK");       // MVAPICH
        if (!lr) lr = getenv("SLURM_LOCALID");                   // SLURM
        if (lr) local_rank = atoi(lr);
        int device = local_rank % ndev;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int world_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        fprintf(stdout,
                "GPU mapping: world_rank=%d local_rank=%d -> device=%d/%d (%s)\n",
                world_rank, local_rank, device, ndev, prop.name);
        fflush(stdout);
    }

    D.set(ni,nj,nk,bc,NS);
    C.NR=NR; C.R=Rconst; C.Ru=Ru; C.P0=P0; C.Rgas_const=Rconst;
    T.myid_x=myid_x; T.myid_y=myid_y; T.myid_z=myid_z;
    T.m_block_x=mbx; T.m_block_y=mby; T.m_block_z=mbz;
    T.m_left=m_left; T.m_right=m_right; T.m_front=m_front;
    T.m_back=m_back; T.m_down=m_down; T.m_up=m_up;
    g_comm = MPI_COMM_WORLD;
    g_dlb_step=dlb_step; g_dlb_tol=dlb_tol;
    if (const char* disabled=getenv("EU3D_DISABLE_DLB"))
        if (strcmp(disabled,"1")==0) g_dlb_step=0;
    int world_rank=0, world_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    // Cross-rank DLB has no work to do in a single-rank run.  Force the
    // single-rank on/off cases through exactly the same fast path so that a
    // nominally enabled DLB setting cannot add a per-step device synchronize.
    if (world_size == 1)
        g_dlb_step = 0;
    g_defer_step_sync = (world_size == 1);
    if (world_rank==0)
        fprintf(stdout,"GPU reaction DLB: %s (step=%d tol=%g); step sync: %s\n",
                world_size==1?"single-rank bypass":
                (g_dlb_step>0?"cross-rank migration enabled":"disabled"),
                g_dlb_step,g_dlb_tol,
                g_defer_step_sync?"deferred":"per-step");

    // Allocate 3D fields.
    F.U=dmalloc(N3());F.V=dmalloc(N3());F.W=dmalloc(N3());F.P=dmalloc(N3());
    F.D=dmalloc(N3());F.T=dmalloc(N3());F.H=dmalloc(N3());F.E=dmalloc(N3());
    F.C=dmalloc(N3());F.Ma=dmalloc(N3());F.Gamma=dmalloc(N3());F.Cp=dmalloc(N3());
    F.Wav=dmalloc(N3());F.Rgas=dmalloc(N3());
    // Allocate 4D fields.
    F.Yi=dmalloc(NSsz());F.Mc=dmalloc(NSsz());F.Di=dmalloc(NSsz());F.CMS=dmalloc(NSsz());
    F.CS=dmalloc(N4());F.F=dmalloc(N4());F.G=dmalloc(N4());F.Q=dmalloc(N4());F.RHS=dmalloc(N4());
    CUDA_CHECK(cudaMalloc(&F.Nchem, N3()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&F.DLBmask, N3()*sizeof(int)));
    CUDA_CHECK(cudaMemset(F.DLBmask, 0, N3()*sizeof(int)));

    // Upload initial fields.
    up(F.U,U,N3());up(F.V,V,N3());up(F.W,W,N3());up(F.P,P,N3());up(F.D,Dd,N3());
    up(F.T,Tt,N3());up(F.H,H,N3());up(F.E,E,N3());up(F.C,Cc,N3());up(F.Ma,Ma,N3());
    up(F.Gamma,Gamma,N3());up(F.Cp,Cp,N3());up(F.Wav,Wav,N3());up(F.Rgas,Rgas,N3());
    up(F.Yi,Yi,NSsz());up(F.Mc,Mc,NSsz());up(F.Di,Di,NSsz());up(F.CS,CS,N4());

    // Upload reaction constants.
    C.Stoi_F=dmalloc((size_t)NS*NR); up(C.Stoi_F,Stoi_F,(size_t)NS*NR);
    C.Stoi_B=dmalloc((size_t)NS*NR); up(C.Stoi_B,Stoi_B,(size_t)NS*NR);
    C.React_TB=dmalloc((size_t)NS*NR); up(C.React_TB,React_TB,(size_t)NS*NR);
    C.Af=dmalloc(NR); up(C.Af,Af,NR);
    C.Bf=dmalloc(NR); up(C.Bf,Bf,NR);
    C.Eaf=dmalloc(NR); up(C.Eaf,Eaf,NR);
    C.Mw=dmalloc(NS); up(C.Mw,Mw,NS);
    C.Ri=dmalloc(NS); up(C.Ri,Ri,NS);
    C.Coeff0=dmalloc((size_t)THERMO_NCOEF*NS); up(C.Coeff0,Coeff0,(size_t)THERMO_NCOEF*NS);
    C.Coeff1=dmalloc((size_t)THERMO_NCOEF*NS); up(C.Coeff1,Coeff1,(size_t)THERMO_NCOEF*NS);
    C.xnode=dmalloc(D.Ni); up(C.xnode,xnode,D.Ni);
    C.ynode=dmalloc(D.Nj); up(C.ynode,ynode,D.Nj);
    C.znode=dmalloc(D.Nk); up(C.znode,znode,D.Nk);

    // Allocate buffers for the largest directional exchange.
    size_t sx=(size_t)D.Nk*D.Nj*D.bc, sy=(size_t)D.Nk*D.Ni*D.bc, sz=(size_t)D.Nj*D.Ni*D.bc;
    size_t smax = sx>sy?(sx>sz?sx:sz):(sy>sz?sy:sz);
    size_t buflen = 9*smax + (size_t)D.NS*smax;
    g_dsend=dmalloc(buflen); g_drecv=dmalloc(buflen);

    g_init = true;
}

/* Advance boundaries, halo exchange, advection, state update, reaction prediction, DLB, reaction and explicit update. */
extern "C" void gpu_solver_step(double dt, int iteration) {
    launch_boundary(F, C, T, D);
    launch_exchange(F, g_dsend, g_drecv, T, D, g_comm);
    launch_advection(F, C, dt, D);
    launch_update_after_adv(F, C, D);          // Add RHS to CS and recover primitive variables.
    launch_trapezoid_prediction(F, C, dt, D);  // Predict Nchem for reaction work estimates.
    launch_trapezoid(F, C, dt, D, iteration, g_dlb_step, g_dlb_tol, g_comm);
    launch_explicit(F, C, D);                  // Rebuild CS from Di and recover primitive variables.
    if (!g_defer_step_sync)
        CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void gpu_solver_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Download fields needed for CPU output. */
extern "C" void gpu_solver_download(
    double* U,double* V,double* W,double* P,double* Dd,double* Tt,
    double* Yi,double* Gamma,double* Ma) {
    down(F.U,U,N3());down(F.V,V,N3());down(F.W,W,N3());down(F.P,P,N3());
    down(F.D,Dd,N3());down(F.T,Tt,N3());down(F.Yi,Yi,NSsz());
    down(F.Gamma,Gamma,N3());down(F.Ma,Ma,N3());
}

// Sum local interior Nchem for load monitoring.
extern "C" long long gpu_solver_sum_nchem() {
    return launch_sum_nchem(F, D);
}

extern "C" int gpu_solver_max_nchem() {
    return launch_max_nchem(F, D);
}

extern "C" void gpu_solver_finalize() {
    if (!g_init) return;
    finalize_advection();
    finalize_exchange();
    finalize_reaction_reduction();
    cudaFree(F.U);cudaFree(F.V);cudaFree(F.W);cudaFree(F.P);cudaFree(F.D);
    cudaFree(F.T);cudaFree(F.H);cudaFree(F.E);cudaFree(F.C);cudaFree(F.Ma);
    cudaFree(F.Gamma);cudaFree(F.Cp);cudaFree(F.Wav);cudaFree(F.Rgas);
    cudaFree(F.Yi);cudaFree(F.Mc);cudaFree(F.Di);cudaFree(F.CMS);
    cudaFree(F.CS);cudaFree(F.F);cudaFree(F.G);cudaFree(F.Q);cudaFree(F.RHS);
    cudaFree(F.Nchem);cudaFree(F.DLBmask);
    cudaFree(C.Stoi_F);cudaFree(C.Stoi_B);cudaFree(C.React_TB);
    cudaFree(C.Af);cudaFree(C.Bf);cudaFree(C.Eaf);cudaFree(C.Mw);cudaFree(C.Ri);
    cudaFree(C.Coeff0);cudaFree(C.Coeff1);cudaFree(C.xnode);cudaFree(C.ynode);cudaFree(C.znode);
    cudaFree(g_dsend);cudaFree(g_drecv);
    g_init=false;
}
