/* MUSCL reconstruction, AUSM fluxes and explicit Euler updates. */
#include "cuda_common.cuh"
#include <cmath>

/* Minmod limiter. */
__device__ __forceinline__ double d_Minmod(double R) {
    if (R > 0.0) return fmin(R, 1.0);
    return 0.0;
}

/* Reconstruct fi into FLR left/right states over [bc-1, n+bc). limit=0 selects Minmod. */
__global__ void kMUSCL2(int direction, const double* __restrict__ fi,
                        double* __restrict__ FLR, Dims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (d.bc - 1);
    int j = blockIdx.y * blockDim.y + threadIdx.y + (d.bc - 1);
    int k = blockIdx.z * blockDim.z + threadIdx.z + (d.bc - 1);
    if (i >= d.ni + d.bc || j >= d.nj + d.bc || k >= d.nk + d.bc) return;

    const double kk = 1.0 / 3.0;
    double A0, A1, A2;

    // Neighbor offsets.
    int di = (direction == 1), dj = (direction == 2), dk = (direction == 3);

    double f_c  = fi[idx3(i,           j,           k,           d.Nj, d.Nk)];
    double f_m1 = fi[idx3(i-di,        j-dj,        k-dk,        d.Nj, d.Nk)];
    double f_p1 = fi[idx3(i+di,        j+dj,        k+dk,        d.Nj, d.Nk)];
    double f_p2 = fi[idx3(i+2*di,      j+2*dj,      k+2*dk,      d.Nj, d.Nk)];

    A0 = f_c  - f_m1;
    A1 = f_p1 - f_c;
    A2 = f_p2 - f_p1;

    double L = A1 / (A0 + 1e-12);
    double R = A1 / (A2 + 1e-12);

    double L1 = d_Minmod(L);
    double L2 = d_Minmod(1.0 / L);
    double R1 = d_Minmod(R);
    double R2 = d_Minmod(1.0 / R);

    int f0 = idx4(i, j, k, 0, d.Nj, d.Nk, 2);
    int f1 = idx4(i, j, k, 1, d.Nj, d.Nk, 2);
    FLR[f0] = f_c  + 0.25 * ((1.0 - kk) * L1 + (1.0 + kk) * L2 * L) * A0;
    FLR[f1] = f_p1 - 0.25 * ((1.0 - kk) * R1 + (1.0 + kk) * R2 * R) * A2;
}

/* AUSM interface fluxes over [1, n+bc). LR fields have shape (Ni,Nj,Nk,2); species fields use NS components. */
__global__ void kAUSM(int direction,
                      const double* __restrict__ PLR, const double* __restrict__ DLR,
                      const double* __restrict__ ULR, const double* __restrict__ VLR,
                      const double* __restrict__ WLR, const double* __restrict__ HLR,
                      const double* __restrict__ GLR,
                      const double* __restrict__ YL_temp, const double* __restrict__ YR_temp,
                      double* __restrict__ Fi, Dims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= d.ni + d.bc || j >= d.nj + d.bc || k >= d.nk + d.bc) return;

    int b0 = idx4(i, j, k, 0, d.Nj, d.Nk, 2);
    int b1 = idx4(i, j, k, 1, d.Nj, d.Nk, 2);

    double PL = PLR[b0], PR = PLR[b1];
    double DL = DLR[b0], DR = DLR[b1];
    double UL = ULR[b0], UR = ULR[b1];
    double VL = VLR[b0], VR = VLR[b1];
    double WL = WLR[b0], WR = WLR[b1];
    double HL = HLR[b0], HR = HLR[b1];
    double GL = GLR[b0], GR = GLR[b1];

    double CL = sqrt(GL * PL / DL);
    double CR = sqrt(GR * PR / DR);
    double CI = 0.5 * (CL + CR);

    double MaL, MaR;
    if (direction == 1)      { MaL = UL / CI; MaR = UR / CI; }
    else if (direction == 2) { MaL = VL / CI; MaR = VR / CI; }
    else                     { MaL = WL / CI; MaR = WR / CI; }

    double B1, B2, G1, G2;
    if (fabs(MaL) < 1.0) {
        double t = (MaL + 1.0);
        double u = (MaL * MaL - 1.0);
        B1 = 0.25 * t * t + 0.125 * u * u;
        G1 = 0.25 * t * t * (2.0 - MaL) + 3.0 / 16.0 * MaL * u * u;
    } else {
        B1 = 0.5 * (MaL + fabs(MaL));
        G1 = B1 / MaL;
    }
    if (fabs(MaR) < 1.0) {
        double t = (MaR - 1.0);
        double u = (MaR * MaR - 1.0);
        B2 = -0.25 * t * t - 0.125 * u * u;
        G2 = 0.25 * t * t * (2.0 + MaR) - 3.0 / 16.0 * MaR * u * u;
    } else {
        B2 = 0.5 * (MaR - fabs(MaR));
        G2 = B2 / MaR;
    }

    double MaI = B1 + B2;
    double VI = MaI * CI;

    // Select the upwind state.
    bool useL = (VI >= 0.0);
    double Dsel = useL ? DL : DR;
    double Usel = useL ? UL : UR;
    double Vsel = useL ? VL : VR;
    double Wsel = useL ? WL : WR;
    double Hsel = useL ? HL : HR;
    const double* Ysel = useL ? YL_temp : YR_temp;

    // Species fluxes.
    for (int s = 0; s < d.NS; s++) {
        double Ys = Ysel[idx4(i, j, k, s, d.Nj, d.Nk, d.NS)];
        Fi[idx4(i, j, k, s, d.Nj, d.Nk, d.NC)] = VI * Dsel * Ys;
    }

    double ke = 0.5 * Dsel * (Usel*Usel + Vsel*Vsel + Wsel*Wsel);
    int NS = d.NS;
    double momU = VI * Dsel * Usel;
    double momV = VI * Dsel * Vsel;
    double momW = VI * Dsel * Wsel;
    double pterm = G1 * PL + G2 * PR;

    // Add pressure to the momentum component normal to the interface.
    Fi[idx4(i, j, k, NS + 0, d.Nj, d.Nk, d.NC)] = momU + (direction == 1 ? pterm : 0.0);
    Fi[idx4(i, j, k, NS + 1, d.Nj, d.Nk, d.NC)] = momV + (direction == 2 ? pterm : 0.0);
    Fi[idx4(i, j, k, NS + 2, d.Nj, d.Nk, d.NC)] = momW + (direction == 3 ? pterm : 0.0);
    Fi[idx4(i, j, k, NS + 3, d.Nj, d.Nk, d.NC)] = VI * (Dsel * Hsel + ke);
}

/* Flux divergence over interior cells [bc, n+bc); dx=xnode(i)-xnode(i-1). */
__global__ void kEE(double dt,
                    const double* __restrict__ xnode,
                    const double* __restrict__ ynode,
                    const double* __restrict__ znode,
                    const double* __restrict__ F,
                    const double* __restrict__ G,
                    const double* __restrict__ Q,
                    double* __restrict__ RHS, Dims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + d.bc;
    int j = blockIdx.y * blockDim.y + threadIdx.y + d.bc;
    int k = blockIdx.z * blockDim.z + threadIdx.z + d.bc;
    if (i >= d.ni + d.bc || j >= d.nj + d.bc || k >= d.nk + d.bc) return;

    double dx = xnode[i] - xnode[i - 1];
    double dy = ynode[j] - ynode[j - 1];
    double dz = znode[k] - znode[k - 1];

    for (int s = 0; s < d.NC; s++) {
        double Fc  = F[idx4(i,   j,   k,   s, d.Nj, d.Nk, d.NC)];
        double Fm  = F[idx4(i-1, j,   k,   s, d.Nj, d.Nk, d.NC)];
        double Gc  = G[idx4(i,   j,   k,   s, d.Nj, d.Nk, d.NC)];
        double Gm  = G[idx4(i,   j-1, k,   s, d.Nj, d.Nk, d.NC)];
        double Qc  = Q[idx4(i,   j,   k,   s, d.Nj, d.Nk, d.NC)];
        double Qm  = Q[idx4(i,   j,   k-1, s, d.Nj, d.Nk, d.NC)];
        RHS[idx4(i, j, k, s, d.Nj, d.Nk, d.NC)] =
            -dt / dx * (Fc - Fm)
            -dt / dy * (Gc - Gm)
            -dt / dz * (Qc - Qm);
    }
}

/* Reconstruct all species in one kernel per direction. */
__global__ void kMUSCL2AllSpecies(int direction,
                                  const double* __restrict__ Yi,
                                  double* __restrict__ YL_temp,
                                  double* __restrict__ YR_temp,
                                  Dims d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (d.bc - 1);
    int j = blockIdx.y * blockDim.y + threadIdx.y + (d.bc - 1);
    int k = blockIdx.z * blockDim.z + threadIdx.z + (d.bc - 1);
    if (i >= d.ni + d.bc || j >= d.nj + d.bc || k >= d.nk + d.bc) return;

    const double kk = 1.0/3.0;
    int di = (direction == 1), dj = (direction == 2), dk = (direction == 3);

    for (int s=0; s<d.NS; s++) {
        double f_c  = Yi[idx4(i,      j,      k,      s,d.Nj,d.Nk,d.NS)];
        double f_m1 = Yi[idx4(i-di,   j-dj,   k-dk,   s,d.Nj,d.Nk,d.NS)];
        double f_p1 = Yi[idx4(i+di,   j+dj,   k+dk,   s,d.Nj,d.Nk,d.NS)];
        double f_p2 = Yi[idx4(i+2*di, j+2*dj, k+2*dk, s,d.Nj,d.Nk,d.NS)];
        double A0=f_c-f_m1, A1=f_p1-f_c, A2=f_p2-f_p1;
        double L=A1/(A0+1e-12), R=A1/(A2+1e-12);
        double L1=d_Minmod(L), L2=d_Minmod(1.0/L);
        double R1=d_Minmod(R), R2=d_Minmod(1.0/R);
        int out=idx4(i,j,k,s,d.Nj,d.Nk,d.NS);
        YL_temp[out] = f_c  + 0.25*((1.0-kk)*L1+(1.0+kk)*L2*L)*A0;
        YR_temp[out] = f_p1 - 0.25*((1.0-kk)*R1+(1.0+kk)*R2*R)*A2;
    }
}

/* Reusable device buffers for reconstructed interface states. */
struct AdvWorkspace {
    double *PLR, *DLR, *ULR, *VLR, *WLR, *HLR, *GLR;
    double *YL_temp, *YR_temp;

    void alloc(const Dims& d) {
        size_t s2 = (size_t)d.Ni * d.Nj * d.Nk * 2 * sizeof(double);
        size_t sNS = (size_t)d.Ni * d.Nj * d.Nk * d.NS * sizeof(double);
        CUDA_CHECK(cudaMalloc(&PLR, s2)); CUDA_CHECK(cudaMalloc(&DLR, s2));
        CUDA_CHECK(cudaMalloc(&ULR, s2)); CUDA_CHECK(cudaMalloc(&VLR, s2));
        CUDA_CHECK(cudaMalloc(&WLR, s2)); CUDA_CHECK(cudaMalloc(&HLR, s2));
        CUDA_CHECK(cudaMalloc(&GLR, s2));
        CUDA_CHECK(cudaMalloc(&YL_temp, sNS));
        CUDA_CHECK(cudaMalloc(&YR_temp, sNS));
    }
    void free_() {
        cudaFree(PLR); cudaFree(DLR); cudaFree(ULR); cudaFree(VLR);
        cudaFree(WLR); cudaFree(HLR); cudaFree(GLR);
        cudaFree(YL_temp); cudaFree(YR_temp);
    }
};

// Launch geometry for interfaces in [1, n+bc).
static inline dim3 blk() { return dim3(8, 8, 4); }
static inline dim3 grdInterface(const Dims& d) {
    dim3 b = blk();
    return dim3((d.ni + d.bc - 1 + b.x - 1) / b.x,
                (d.nj + d.bc - 1 + b.y - 1) / b.y,
                (d.nk + d.bc - 1 + b.z - 1) / b.z);
}
/* Reconstruct seven scalar fields and NS species, then compute directional AUSM fluxes. */
static void ausm_dir(int direction, const Dims& d, AdvWorkspace& w,
                     const double* P, const double* D, const double* U,
                     const double* V, const double* W, const double* H,
                     const double* Gamma, const double* Yi, double* Fi) {
    dim3 b = blk(), g = grdInterface(d);

    kMUSCL2<<<g, b>>>(direction, P,     w.PLR, d);
    kMUSCL2<<<g, b>>>(direction, D,     w.DLR, d);
    kMUSCL2<<<g, b>>>(direction, U,     w.ULR, d);
    kMUSCL2<<<g, b>>>(direction, V,     w.VLR, d);
    kMUSCL2<<<g, b>>>(direction, W,     w.WLR, d);
    kMUSCL2<<<g, b>>>(direction, H,     w.HLR, d);
    kMUSCL2<<<g, b>>>(direction, Gamma, w.GLR, d);

    kMUSCL2AllSpecies<<<g,b>>>(direction, Yi, w.YL_temp, w.YR_temp, d);

    kAUSM<<<g, b>>>(direction, w.PLR, w.DLR, w.ULR, w.VLR, w.WLR, w.HLR, w.GLR,
                    w.YL_temp, w.YR_temp, Fi, d);
}

/* Advection update; all pointers are device pointers. The caller allocates F/G/Q/RHS. */
extern "C" void advection_gpu(const Dims& d, double dt,
                              const double* xnode, const double* ynode, const double* znode,
                              const double* P, const double* D, const double* U,
                              const double* V, const double* W, const double* H,
                              const double* Gamma, const double* Yi,
                              double* F, double* G, double* Q, double* RHS,
                              AdvWorkspace& w) {
    size_t sizeNC = (size_t)d.Ni * d.Nj * d.Nk * d.NC * sizeof(double);
    CUDA_CHECK(cudaMemset(F, 0, sizeNC));
    CUDA_CHECK(cudaMemset(G, 0, sizeNC));
    CUDA_CHECK(cudaMemset(Q, 0, sizeNC));

    ausm_dir(1, d, w, P, D, U, V, W, H, Gamma, Yi, F);
    ausm_dir(2, d, w, P, D, U, V, W, H, Gamma, Yi, G);
    ausm_dir(3, d, w, P, D, U, V, W, H, Gamma, Yi, Q);

    dim3 b = blk(), g = grdInterface(d);
    kEE<<<g, b>>>(dt, xnode, ynode, znode, F, G, Q, RHS, d);
    CUDA_CHECK(cudaGetLastError());
}

/* C wrappers hide AdvWorkspace from callers. */
extern "C" void* adv_ws_alloc(const Dims& d) {
    AdvWorkspace* w = new AdvWorkspace();
    w->alloc(d);
    return (void*)w;
}

extern "C" void adv_ws_free(void* wp) {
    AdvWorkspace* w = (AdvWorkspace*)wp;
    w->free_();
    delete w;
}

extern "C" void advection_gpu_v(const Dims& d, double dt,
                                const double* xnode, const double* ynode, const double* znode,
                                const double* P, const double* D, const double* U,
                                const double* V, const double* W, const double* H,
                                const double* Gamma, const double* Yi,
                                double* F, double* G, double* Q, double* RHS, void* wp) {
    AdvWorkspace* w = (AdvWorkspace*)wp;
    advection_gpu(d, dt, xnode, ynode, znode, P, D, U, V, W, H, Gamma, Yi,
                  F, G, Q, RHS, *w);
}

/* Host launcher. */
#include "kernels.cuh"

// One persistent workspace per process.
static AdvWorkspace* g_advws = nullptr;

void launch_advection(DevFields& f, const DevConst& c, double dt, const Dims& d) {
    if (!g_advws) { g_advws = new AdvWorkspace(); g_advws->alloc(d); }
    advection_gpu(d, dt, c.xnode, c.ynode, c.znode,
                  f.P, f.D, f.U, f.V, f.W, f.H, f.Gamma, f.Yi,
                  f.F, f.G, f.Q, f.RHS, *g_advws);
}

void finalize_advection() {
    if (!g_advws) return;
    g_advws->free_();
    delete g_advws;
    g_advws = nullptr;
}
