/* Adaptive trapezoidal reaction integration, one cell per thread. Row-major constants: Stoi_F/Stoi_B/React_TB=(NS,NR), Af/Bf/Eaf=(NR), Coeff0/Coeff1=(9,NS). Local arrays require NS<=16 and NR<=24. */
#include "cuda_common.cuh"
#include "thermo.cuh"
#include "dlb_schedule.hpp"
#include "warp_queue.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define MAX_NS 16
#define MAX_NR 24

struct ReactConst {
    int NR;
    double R, Ru, P0;
    const double *Stoi_F, *Stoi_B, *React_TB;   // (NS,NR)
    const double *Af, *Bf, *Eaf;                // (NR)
    const double *Mw;                           // (NS)
    const double *Coeff0, *Coeff1;              // (9,NS)
};

/* Compute net rates, third-body corrections and forward/reverse rate constants. */
__device__ __forceinline__
void computeRates(double Tl, const double* Mc_cell, const ReactConst& rc, int NS,
                  double* KF, double* KB, double* RR_F, double* RR_B,
                  double* R_TB, double* RR, double* Gi) {
    int NR = rc.NR;
    for (int s=0;s<NS;s++) {
        double Hi = d_GetHi(Tl, rc.R, s, rc.Coeff0, rc.Coeff1, NS);
        double Si = d_GetSi(Tl, rc.R, s, rc.Coeff0, rc.Coeff1, NS);
        Gi[s] = Hi - Si*Tl;
    }
    for (int r=0;r<NR;r++){ RR_F[r]=1.0; RR_B[r]=1.0; R_TB[r]=0.0; }

    for (int r=0;r<NR;r++) {
        double Xr=0.0, delta_g=0.0;
        KF[r] = rc.Af[r] * pow(Tl, rc.Bf[r]) * exp(-rc.Eaf[r]/(rc.Ru*Tl));
        for (int s=0;s<NS;s++) {
            double sf = rc.Stoi_F[s*NR+r];
            double sb = rc.Stoi_B[s*NR+r];
            Xr      += (sb - sf);
            delta_g += (sb - sf) * Gi[s];
            RR_F[r] *= pow(Mc_cell[s], sf);
            RR_B[r] *= pow(Mc_cell[s], sb);
            R_TB[r] += Mc_cell[s] * rc.React_TB[s*NR+r];
        }
        double Kp = exp(-delta_g/(rc.R*Tl)) * pow(rc.P0*1e-6, Xr);
        double Kc = Kp * pow(rc.R*Tl, -Xr);
        KB[r] = KF[r]/Kc;
        if (R_TB[r]==0.0) R_TB[r]=1.0;
        RR[r] = KF[r]*RR_F[r] - KB[r]*RR_B[r];
    }
}

/* Predict Nchem from per-cell reaction rates. */
__global__ void kTrapezoidPrediction(const double* __restrict__ Mc_temp,
                                     const double* __restrict__ Di_temp,
                                     const double* __restrict__ Yi,
                                     const double* __restrict__ T,
                                     int* __restrict__ Nchem,
                                     double dtm, ReactConst rc, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int NS=d.NS, NR=rc.NR;

    double Tl = T[idx3(i,j,k,d.Nj,d.Nk)];
    double Di_cell[MAX_NS], Mc_cell[MAX_NS];
    double KF[MAX_NR],KB[MAX_NR],RR_F[MAX_NR],RR_B[MAX_NR],R_TB[MAX_NR],RR[MAX_NR],Gi[MAX_NS];

    for (int s=0;s<NS;s++) {
        Di_cell[s] = Di_temp[idx4(i,j,k,s,d.Nj,d.Nk,NS)] * 1e-3;
        Mc_cell[s] = Mc_temp[idx4(i,j,k,s,d.Nj,d.Nk,NS)] * 1e-6;
    }
    computeRates(Tl, Mc_cell, rc, NS, KF,KB,RR_F,RR_B,R_TB,RR,Gi);

    double dt = dtm;
    for (int s=0;s<NS;s++) {
        double temp=0.0;
        for (int r=0;r<NR;r++)
            temp += (rc.Stoi_B[s*NR+r]-rc.Stoi_F[s*NR+r]) * R_TB[r] * RR[r];
        double Wi = rc.Mw[s]*temp;
        double y = Yi[idx4(i,j,k,s,d.Nj,d.Nk,NS)];
        if (y >= 1e-6 && Wi != 0.0) {
            double dt_temp = fabs(-Di_cell[s]/Wi);
            if (dt_temp < dt) dt = dt_temp;
        }
    }
    int nc = (int)ceil(dtm/dt);
    if (nc < 1) nc = 1;
    Nchem[idx3(i,j,k,d.Nj,d.Nk)] = nc;
}

/* Integrate Di over Nchem substeps. Recompute rates from Mc/T, advance species by dtm/Nchem, then update Yi/Mc/T. Velocity and E remain fixed during chemistry. */
__device__ __forceinline__
void solveReactionCell(double* Di_cell, double* Mc_cell, double& Tl, double energy,
                       int nc, double dtm, double* CMS_cell,
                       const double* Ri, ReactConst rc, int NS) {
    int NR=rc.NR;
    double dt = dtm / nc;
    double KF[MAX_NR],KB[MAX_NR],RR_F[MAX_NR],RR_B[MAX_NR],R_TB[MAX_NR],RR[MAX_NR],Gi[MAX_NS];
    double P[MAX_NS], Q[MAX_NS];

    for (int step=0; step<nc; step++) {
        computeRates(Tl, Mc_cell, rc, NS, KF,KB,RR_F,RR_B,R_TB,RR,Gi);
        for (int s=0;s<NS;s++) {
            double temp=0.0;
            for (int r=0;r<NR;r++)
                temp += (rc.Stoi_B[s*NR+r]-rc.Stoi_F[s*NR+r]) * R_TB[r] * RR[r];
            CMS_cell[s] = rc.Mw[s]*temp*1e3;
        }
        for (int s=0;s<NS;s++){ P[s]=0.0; Q[s]=0.0; }
        for (int s=0;s<NS;s++) {
            for (int r=0;r<NR;r++) {
                Q[s] += (rc.Stoi_B[s*NR+r]*KF[r]*RR_F[r] + rc.Stoi_F[s*NR+r]*KB[r]*RR_B[r]) * R_TB[r];
                P[s] += (rc.Stoi_B[s*NR+r]*KB[r]*RR_B[r] + rc.Stoi_F[s*NR+r]*KF[r]*RR_F[r]) * R_TB[r];
            }
            Q[s] = rc.Mw[s]*Q[s];
            P[s] = (Mc_cell[s]==0.0) ? 0.0 : P[s]/Mc_cell[s];
        }
        double dens=0.0;
        for (int s=0;s<NS;s++) {
            Di_cell[s] = ((1.0 - dt/2.0*P[s])*Di_cell[s] + dt*Q[s])
                         / (1.0 + dt/2.0*P[s]);
            dens += Di_cell[s];
        }
        for (int s=0;s<NS;s++) {
            Q[s] = Di_cell[s]/dens;
            Mc_cell[s] = Di_cell[s]/rc.Mw[s];
        }
        Tl = d_GetTemp(Tl, Q, Ri, energy, rc.Coeff0, rc.Coeff1, NS);
    }
}

__global__ void kTrapezoid(const double* __restrict__ Mc_temp,
                          double* __restrict__ Di_temp,
                          double* __restrict__ T,
                          const double* __restrict__ E,
                          const int* __restrict__ Nchem,
                          const int* __restrict__ DLBmask,
                          double* __restrict__ CMS,
                          const double* __restrict__ Ri,
                          double dtm, ReactConst rc, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int NS=d.NS;

    int cell = idx3(i,j,k,d.Nj,d.Nk);
    if (DLBmask && DLBmask[cell]) return;
    double Tl = T[cell];
    int nc = Nchem[cell];

    double Di_cell[MAX_NS], Mc_cell[MAX_NS];
    double CMS_cell[MAX_NS];

    for (int s=0;s<NS;s++) {
        Di_cell[s] = Di_temp[idx4(i,j,k,s,d.Nj,d.Nk,NS)] * 1e-3;
        Mc_cell[s] = Mc_temp[idx4(i,j,k,s,d.Nj,d.Nk,NS)] * 1e-6;
    }

    solveReactionCell(Di_cell, Mc_cell, Tl, E[cell], nc, dtm, CMS_cell, Ri, rc, NS);

    T[cell] = Tl;
    for (int s=0;s<NS;s++) {
        Di_temp[idx4(i,j,k,s,d.Nj,d.Nk,NS)] = Di_cell[s]*1e3;
        CMS[idx4(i,j,k,s,d.Nj,d.Nk,NS)] = CMS_cell[s];
    }
}

/*---------------------------------------------------------------------------------
 * Intra-GPU warp-level load balancing.
 *
 * kTrapezoid maps one thread to one grid point using an 8x8x4 block, so a warp
 * covers an 8(i) x 4(j) x 1(k) tile.  The detonation front cuts across such a
 * tile, which means Nchem inside a single warp ranges from 1 to a few hundred and
 * every thread has to wait for the longest one.  The instrumented 8-GPU run
 * measured 66.4% warp waste on the front-carrying rank while cross-rank DLB left
 * that number untouched.
 *
 * The fix is a counting sort of the interior cells into Nchem buckets followed by
 * a flat list kernel, so each warp receives 32 cells of similar iteration count.
 * Cells inside a bucket are placed in atomic order, which is non-deterministic,
 * but every cell's reaction only reads and writes its own location, so the
 * numerical result is independent of the ordering and stays bit-identical.
 *---------------------------------------------------------------------------------*/
#define EU3D_NCHEM_BUCKETS 8

__host__ __device__ __forceinline__ int nchem_bucket(int nc) {
    if (nc <= 1)   return 0;
    if (nc <= 4)   return 1;
    if (nc <= 9)   return 2;
    if (nc <= 19)  return 3;
    if (nc <= 49)  return 4;
    if (nc <= 99)  return 5;
    if (nc <= 199) return 6;
    return 7;
}

__device__ __forceinline__ int d_compact_to_cell(int q, const Dims& d) {
    int k=q%d.nk;
    int t=q/d.nk;
    int j=t%d.nj;
    int i=t/d.nj;
    return idx3(i+d.bc,j+d.bc,k+d.bc,d.Nj,d.Nk);
}

/* One global atomic per bucket per block instead of one per cell. */
__global__ void kCountNchemBuckets(const int* __restrict__ Nchem,
                                   const int* __restrict__ DLBmask,
                                   int* __restrict__ counts, Dims d) {
    __shared__ int s_count[EU3D_NCHEM_BUCKETS];
    if (threadIdx.x < EU3D_NCHEM_BUCKETS) s_count[threadIdx.x]=0;
    __syncthreads();

    const int total=d.ni*d.nj*d.nk;
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q<total) {
        int cell=d_compact_to_cell(q,d);
        if (!(DLBmask && DLBmask[cell]))
            atomicAdd(&s_count[nchem_bucket(Nchem[cell])],1);
    }
    __syncthreads();

    if (threadIdx.x < EU3D_NCHEM_BUCKETS && s_count[threadIdx.x])
        atomicAdd(&counts[threadIdx.x],s_count[threadIdx.x]);
}

/* Exclusive scan over 8 buckets; trivial enough to keep on one thread and it
 * avoids a host round trip in the middle of the timestep. */
__global__ void kNchemBucketOffsets(const int* __restrict__ counts,
                                    int* __restrict__ offsets,
                                    int* __restrict__ cursors) {
    if (blockIdx.x || threadIdx.x) return;
    int run=0;
    for (int b=0;b<EU3D_NCHEM_BUCKETS;b++) {
        offsets[b]=run;
        cursors[b]=run;
        run+=counts[b];
    }
    offsets[EU3D_NCHEM_BUCKETS]=run;
}

__global__ void kScatterNchemBuckets(const int* __restrict__ Nchem,
                                     const int* __restrict__ DLBmask,
                                     int* __restrict__ cursors,
                                     int* __restrict__ cell_list, Dims d) {
    __shared__ int s_count[EU3D_NCHEM_BUCKETS];
    __shared__ int s_cursor[EU3D_NCHEM_BUCKETS];
    if (threadIdx.x < EU3D_NCHEM_BUCKETS) s_count[threadIdx.x]=0;
    __syncthreads();

    const int total=d.ni*d.nj*d.nk;
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int cell=-1, bucket=-1;
    if (q<total) {
        cell=d_compact_to_cell(q,d);
        if (!(DLBmask && DLBmask[cell])) {
            bucket=nchem_bucket(Nchem[cell]);
            atomicAdd(&s_count[bucket],1);
        }
    }
    __syncthreads();

    // Reserve one contiguous run per bucket for the whole block.
    if (threadIdx.x < EU3D_NCHEM_BUCKETS)
        s_cursor[threadIdx.x]= s_count[threadIdx.x]
            ? atomicAdd(&cursors[threadIdx.x],s_count[threadIdx.x])
            : 0;
    __syncthreads();

    if (bucket>=0)
        cell_list[atomicAdd(&s_cursor[bucket],1)]=cell;
}

/* Same numerics as kTrapezoid, driven by an explicit cell list.  The DLB mask is
 * already applied while building the list, so masked cells never appear here. */
__global__ void kTrapezoidList(const double* __restrict__ Mc_temp,
                               double* __restrict__ Di_temp,
                               double* __restrict__ T,
                               const double* __restrict__ E,
                               const int* __restrict__ Nchem,
                               const int* __restrict__ cell_list,
                               const int* __restrict__ list_len,
                               double* __restrict__ CMS,
                               const double* __restrict__ Ri,
                               double dtm, ReactConst rc, Dims d) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=list_len[EU3D_NCHEM_BUCKETS]) return;
    const int NS=d.NS;
    const int cell=cell_list[q];

    double Tl=T[cell];
    int nc=Nchem[cell];

    double Di_cell[MAX_NS], Mc_cell[MAX_NS];
    double CMS_cell[MAX_NS];

    for (int s=0;s<NS;s++) {
        Di_cell[s] = Di_temp[(size_t)cell*NS+s] * 1e-3;
        Mc_cell[s] = Mc_temp[(size_t)cell*NS+s] * 1e-6;
    }

    solveReactionCell(Di_cell, Mc_cell, Tl, E[cell], nc, dtm, CMS_cell, Ri, rc, NS);

    T[cell] = Tl;
    for (int s=0;s<NS;s++) {
        Di_temp[(size_t)cell*NS+s] = Di_cell[s]*1e3;
        CMS[(size_t)cell*NS+s] = CMS_cell[s];
    }
}

__global__ void kTrapezoidWarpQueue(const double* __restrict__ Mc_temp,
                               double* __restrict__ Di_temp,
                               double* __restrict__ T,
                               const double* __restrict__ E,
                               const int* __restrict__ Nchem,
                               const int* __restrict__ cell_list,
                               const int* __restrict__ list_len,
                               int* __restrict__ queue_heads,
                               double* __restrict__ CMS,
                               const double* __restrict__ Ri,
                               double dtm, ReactConst rc, Dims d) {
    int bucket=EU3D_NCHEM_BUCKETS-1;
    int begin=0, end=0;
    while (warp_queue_claim(list_len,queue_heads,bucket,begin,end)) {
        const int q=begin+(threadIdx.x&31);
        if (q<end) {
            const int NS=d.NS;
            const int cell=cell_list[q];

            double Tl=T[cell];
            int nc=Nchem[cell];

            double Di_cell[MAX_NS], Mc_cell[MAX_NS];
            double CMS_cell[MAX_NS];

            for (int s=0;s<NS;s++) {
                Di_cell[s] = Di_temp[(size_t)cell*NS+s] * 1e-3;
                Mc_cell[s] = Mc_temp[(size_t)cell*NS+s] * 1e-6;
            }

            solveReactionCell(Di_cell, Mc_cell, Tl, E[cell], nc, dtm, CMS_cell, Ri, rc, NS);

            T[cell] = Tl;
            for (int s=0;s<NS;s++) {
                Di_temp[(size_t)cell*NS+s] = Di_cell[s]*1e3;
                CMS[(size_t)cell*NS+s] = CMS_cell[s];
            }
        }
        __syncwarp(0xffffffffu);
    }
}

/* Cross-rank DLB task format:
 * input  = [Nchem, T, E, Di[NS], Mc[NS]]
 * output = [T, Di[NS], CMS[NS]]
 * Cell indices stay on the owning rank; returned records preserve send order. */
__global__ void kSetDLBMask(int* mask, const int* cells, int count) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q<count) mask[cells[q]]=1;
}

__global__ void kCompactNchem(const int* Nchem, int* compact, Dims d) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int count=d.ni*d.nj*d.nk;
    if (q>=count) return;
    int k=q%d.nk;
    int t=q/d.nk;
    int j=t%d.nj;
    int i=t/d.nj;
    int cell=idx3(i+d.bc,j+d.bc,k+d.bc,d.Nj,d.Nk);
    compact[q]=Nchem[cell];
}

__global__ void kPackReactionTasks(const int* cells, int count,
                                   const int* Nchem, const double* T, const double* E,
                                   const double* Di, const double* Mc,
                                   double* records, int NS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=count) return;
    int cell=cells[q], stride=2*NS+3;
    double* out=records+(size_t)q*stride;
    out[0]=(double)Nchem[cell]; out[1]=T[cell]; out[2]=E[cell];
    for (int s=0;s<NS;s++) {
        out[3+s]=Di[(size_t)cell*NS+s];
        out[3+NS+s]=Mc[(size_t)cell*NS+s];
    }
}

__global__ void kSolveReactionTasks(const double* input, double* output, int count,
                                    double dtm, const double* Ri,
                                    ReactConst rc, int NS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=count) return;
    int in_stride=2*NS+3, out_stride=2*NS+1;
    const double* in=input+(size_t)q*in_stride;
    double* out=output+(size_t)q*out_stride;
    int nc=(int)in[0];
    double Tl=in[1], energy=in[2];
    double Di_cell[MAX_NS],Mc_cell[MAX_NS],CMS_cell[MAX_NS];
    for (int s=0;s<NS;s++) {
        Di_cell[s]=in[3+s]*1e-3;
        Mc_cell[s]=in[3+NS+s]*1e-6;
    }
    solveReactionCell(Di_cell,Mc_cell,Tl,energy,nc,dtm,CMS_cell,Ri,rc,NS);
    out[0]=Tl;
    for (int s=0;s<NS;s++) {
        out[1+s]=Di_cell[s]*1e3;
        out[1+NS+s]=CMS_cell[s];
    }
}

__global__ void kUnpackReactionTasks(const int* cells, int count,
                                     const double* records,
                                     double* T, double* Di, double* CMS, int NS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=count) return;
    int cell=cells[q], stride=2*NS+1;
    const double* in=records+(size_t)q*stride;
    T[cell]=in[0];
    for (int s=0;s<NS;s++) {
        Di[(size_t)cell*NS+s]=in[1+s];
        CMS[(size_t)cell*NS+s]=in[1+NS+s];
    }
}

/* Recover primitive variables from conserved state CS. */
__global__ void kUpdateAfterCS(const double* __restrict__ CS,
                              double* __restrict__ D, double* __restrict__ U,
                              double* __restrict__ V, double* __restrict__ W,
                              double* __restrict__ E, double* __restrict__ P,
                              double* __restrict__ H, double* __restrict__ T,
                              double* __restrict__ Gamma, double* __restrict__ C,
                              double* __restrict__ Ma,
                              double* __restrict__ Yi, double* __restrict__ Di,
                              double* __restrict__ Mc,
                              const double* __restrict__ Mw, const double* __restrict__ Ri,
                              const double* __restrict__ Coeff0, const double* __restrict__ Coeff1,
                              double Rgas_const, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int NS=d.NS, NC=d.NC, c=idx3(i,j,k,d.Nj,d.Nk);

    double Yc[MAX_NS];
    double dens=0.0;
    for (int s=0;s<NS;s++){
        double cs = CS[idx4(i,j,k,s,d.Nj,d.Nk,NC)];
        dens += cs;
        Di[idx4(i,j,k,s,d.Nj,d.Nk,NS)] = cs;
        Mc[idx4(i,j,k,s,d.Nj,d.Nk,NS)] = cs/Mw[s]*1000.0;
    }
    D[c]=dens;
    for (int s=0;s<NS;s++){
        double y = CS[idx4(i,j,k,s,d.Nj,d.Nk,NC)]/dens;
        Yi[idx4(i,j,k,s,d.Nj,d.Nk,NS)] = y;
        Yc[s]=y;
    }
    double u = CS[idx4(i,j,k,NS+0,d.Nj,d.Nk,NC)]/dens;
    double v = CS[idx4(i,j,k,NS+1,d.Nj,d.Nk,NC)]/dens;
    double w = CS[idx4(i,j,k,NS+2,d.Nj,d.Nk,NC)]/dens;
    double etot = CS[idx4(i,j,k,NS+3,d.Nj,d.Nk,NC)];
    double e = (etot - 0.5*dens*(u*u+v*v+w*w))/dens;

    double T_new = d_GetTemp(T[c], Yc, Ri, e, Coeff0, Coeff1, NS);
    T[c]=T_new; U[c]=u; V[c]=v; W[c]=w; E[c]=e;

    double sumYMw=0.0;                     // Fun.sum(3,Yi,Mw) = sum(Yi/Mw)
    for (int s=0;s<NS;s++) sumYMw += Yc[s]/Mw[s];
    double Wav = 1.0/sumYMw;
    double Rgas = Rgas_const*1000.0/Wav;   // R=8.31434
    double p = T_new*dens*Rgas;
    P[c]=p; H[c]=p/dens + e;

    double Cp=0.0;
    for (int s=0;s<NS;s++) Cp += Yc[s]*d_GetCpi(T_new, Ri[s], s, Coeff0, Coeff1, NS);
    Cp *= 1000.0;
    double gamma = Cp/(Cp - Rgas_const*sumYMw*1000.0);
    Gamma[c]=gamma;
    double cc = sqrt(gamma*p/dens);
    C[c]=cc;
    Ma[c]=sqrt(u*u+v*v+w*w)/cc;   // Match the CPU Mach-number update.
}

/* Add the advection RHS to CS. */
__global__ void kUpdateAfterAdv(double* __restrict__ CS, const double* __restrict__ RHS,
                               Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    for (int s=0;s<d.NC;s++) {
        int idx = idx4(i,j,k,s,d.Nj,d.Nk,d.NC);
        CS[idx] += RHS[idx];
    }
}

/* Rebuild conserved species, momentum and energy from Di, then recover primitive variables. */
__global__ void kExplicit(double* __restrict__ CS, const double* __restrict__ Di,
                         double* __restrict__ D,
                         const double* __restrict__ U, const double* __restrict__ V,
                         const double* __restrict__ W, const double* __restrict__ E,
                         Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int NS=d.NS, NC=d.NC, c=idx3(i,j,k,d.Nj,d.Nk);

    double dens=0.0;
    for (int s=0;s<NS;s++){
        double di = Di[idx4(i,j,k,s,d.Nj,d.Nk,NS)];
        CS[idx4(i,j,k,s,d.Nj,d.Nk,NC)] = di;
        dens += di;
    }
    D[c]=dens;
    double u=U[c], v=V[c], w=W[c], e=E[c];
    CS[idx4(i,j,k,NS+0,d.Nj,d.Nk,NC)] = u*dens;
    CS[idx4(i,j,k,NS+1,d.Nj,d.Nk,NC)] = v*dens;
    CS[idx4(i,j,k,NS+2,d.Nj,d.Nk,NC)] = w*dens;
    CS[idx4(i,j,k,NS+3,d.Nj,d.Nk,NC)] = dens*e + 0.5*(u*u+v*v+w*w)*dens;
}

/* Host launchers. */
#include "kernels.cuh"

static inline dim3 rblk() { return dim3(8,8,4); }
static inline dim3 rgrdInner(const Dims& d) {
    dim3 b=rblk();
    return dim3((d.ni+b.x-1)/b.x,(d.nj+b.y-1)/b.y,(d.nk+b.z-1)/b.z);
}

static ReactConst makeRC(const DevConst& c) {
    ReactConst rc;
    rc.NR=c.NR; rc.R=c.R; rc.Ru=c.Ru; rc.P0=c.P0;
    rc.Stoi_F=c.Stoi_F; rc.Stoi_B=c.Stoi_B; rc.React_TB=c.React_TB;
    rc.Af=c.Af; rc.Bf=c.Bf; rc.Eaf=c.Eaf; rc.Mw=c.Mw;
    rc.Coeff0=c.Coeff0; rc.Coeff1=c.Coeff1;
    return rc;
}

long long launch_sum_nchem(DevFields& f, const Dims& d);

namespace {
std::vector<int> g_out_cells, g_send_tasks, g_recv_tasks;
int *g_d_compact=nullptr, *g_d_cells=nullptr;
double *g_d_task_in=nullptr, *g_d_task_recv=nullptr;
double *g_d_task_out=nullptr, *g_d_task_return=nullptr;
size_t g_compact_cap=0, g_cells_cap=0, g_task_in_cap=0, g_task_recv_cap=0;
size_t g_task_out_cap=0, g_task_return_cap=0;
double *g_h_task_send=nullptr,*g_h_task_recv=nullptr,*g_h_task_done=nullptr,*g_h_task_return=nullptr;
size_t g_h_send_cap=0,g_h_recv_cap=0,g_h_done_cap=0,g_h_return_cap=0;
bool g_plan_ready=false, g_plan_has_tasks=false;

/* Intra-GPU bucket scheduling state. */
int *g_d_bucket_counts=nullptr;    // EU3D_NCHEM_BUCKETS
int *g_d_bucket_offsets=nullptr;   // EU3D_NCHEM_BUCKETS+1, [last] = list length
int *g_d_bucket_cursors=nullptr;   // EU3D_NCHEM_BUCKETS
int *g_d_cell_list=nullptr;        // interior cell count
size_t g_cell_list_cap=0;
int* g_d_queue_heads=nullptr;
int g_queue_blocks=0;
long long g_queue_launches=0;
bool g_bucket_ready=false;         // buffers allocated
bool g_bucket_useful=true;         // refreshed on the replan cadence
long long g_bucket_evaluations=0, g_bucket_bypasses=0;

/* CUDA streams.  Stream 0 (legacy default) keeps carrying every non-DLB kernel so
 * ordering with advection/boundary/update is untouched.  The DLB stream is created
 * non-blocking on purpose: a blocking stream implicitly synchronises with the
 * legacy default stream, which would defeat the whole point of overlapping the
 * migrated reaction work with the local reaction kernel. */
cudaStream_t g_dlb_stream=nullptr;
cudaEvent_t g_ev_inputs_ready=nullptr;   // default -> dlb  (Nchem/T/E/Di/Mc valid)
cudaEvent_t g_ev_dlb_done=nullptr;       // dlb -> default  (unpack finished)

struct DlbTiming {
    double plan=0.0, pack=0.0, forward=0.0, remote=0.0, backward=0.0, unpack=0.0;
    // Probe timers: separate rank arrival skew from the actual data transfer.
    double forward_skew=0.0, forward_transfer=0.0;
    double backward_skew=0.0, backward_transfer=0.0;
    long long replans=0, skipped_replans=0, active_steps=0, planned_tasks=0;
    long long forward_bytes=0, backward_bytes=0;
    long long forward_messages=0, backward_messages=0;
    long long gate_rejected=0, budget_stops=0;
};
DlbTiming g_dlb_timing;

bool env_flag(const char* name) {
    const char* value=std::getenv(name);
    return value && std::strcmp(value,"1")==0;
}

bool env_equals(const char* name, const char* expected) {
    const char* value=std::getenv(name);
    return value && std::strcmp(value,expected)==0;
}

long env_int(const char* name, long fallback) {
    const char* value=std::getenv(name);
    if (!value || !*value) return fallback;
    char* end=nullptr;
    long parsed=std::strtol(value,&end,10);
    if (end==value) return fallback;
    return parsed;
}

bool dlb_cuda_aware_requested() {
    static const bool requested=env_flag("EU3D_CUDA_AWARE_MPI");
    return requested;
}

/* EU3D_DLB_SCHEDULE=overlap
 * Reorders the DLB timestep so the forward exchange runs while the GPU is idle
 * and the migrated reaction solve runs concurrently with the local one. */
bool dlb_overlap_requested() {
    static const bool requested=env_equals("EU3D_DLB_SCHEDULE","overlap");
    return requested;
}

/* EU3D_DLB_PROBE=1
 * Adds an MPI_Barrier immediately before each exchange so the recorded MPI time
 * can be split into "waiting for peers to arrive" and "moving bytes". */
bool dlb_probe_requested() {
    static const bool requested=env_flag("EU3D_DLB_PROBE");
    return requested;
}

/* EU3D_DLB_TASK_ORDER=nchem
 * Orders each peer's migrated task list by descending Nchem instead of by cell
 * address, so the remote solve kernel gets warp-uniform iteration counts. */
bool dlb_task_order_nchem() {
    static const bool requested=env_equals("EU3D_DLB_TASK_ORDER","nchem");
    return requested;
}

bool reaction_sched_dynamic() {
    static const bool requested=env_equals("EU3D_REACTION_SCHED","dynamic");
    return requested;
}

/* EU3D_REACTION_SCHED=baseline|sorted|dynamic */
bool reaction_sched_sorted() {
    static const bool requested=env_equals("EU3D_REACTION_SCHED","sorted");
    return requested || reaction_sched_dynamic();
}

/* EU3D_DLB_COST_GATE=1 plus the two thresholds it controls. */
bool dlb_cost_gate_requested() {
    static const bool requested=env_flag("EU3D_DLB_COST_GATE");
    return requested;
}

int dlb_min_task_weight() {
    static const int weight=(int)env_int("EU3D_DLB_MIN_TASK_WEIGHT",10);
    return weight;
}

long long dlb_max_tasks_per_replan() {
    static const long long budget=(long long)env_int("EU3D_DLB_MAX_TASKS",0);
    return budget;
}

/* Minimum fraction of cells that must sit outside bucket 0 before the counting
 * sort is worth its two extra passes.  Ranks 4-7 of the 8-GPU case measured
 * Nchem==1 for every single cell and warp waste of exactly 0%, so for them the
 * bucket path is pure overhead and must be bypassed. */
double reaction_sched_min_heavy_fraction() {
    static const double fraction=[]() {
        const char* value=std::getenv("EU3D_REACTION_SCHED_MIN_HEAVY");
        if (!value || !*value) return 1e-4;
        char* end=nullptr;
        double parsed=std::strtod(value,&end);
        return end==value?1e-4:parsed;
    }();
    return fraction;
}

template <class T>
void ensure_device(T*& ptr, size_t& capacity, size_t needed) {
    if (capacity>=needed) return;
    if (ptr) CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(needed,1)*sizeof(T)));
    capacity=std::max<size_t>(needed,1);
}

void ensure_pinned(double*& ptr,size_t& capacity,size_t needed) {
    if (capacity>=needed) return;
    if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
    CUDA_CHECK(cudaMallocHost((void**)&ptr,std::max<size_t>(needed,1)*sizeof(double)));
    capacity=std::max<size_t>(needed,1);
}

int compact_to_cell(int q, const Dims& d) {
    int k=q%d.nk;
    int t=q/d.nk;
    int j=t%d.nj;
    int i=t/d.nj;
    return idx3(i+d.bc,j+d.bc,k+d.bc,d.Nj,d.Nk);
}

/* Separate posting from waiting to allow work between MPI calls. */
struct PeerExchange {
    std::vector<MPI_Request> requests;
    long long bytes=0;
    long long messages=0;
};

void post_exchange_active_peers(PeerExchange& ex,
                                const double* sendbuf,
                                const std::vector<int>& sendcounts,
                                const std::vector<int>& sdispls,
                                double* recvbuf,
                                const std::vector<int>& recvcounts,
                                const std::vector<int>& rdispls,
                                int tag, MPI_Comm comm) {
    int nranks=1;
    MPI_Comm_size(comm,&nranks);
    ex.requests.clear();
    ex.requests.reserve((size_t)2*nranks);
    ex.bytes=0;
    ex.messages=0;
    for (int r=0;r<nranks;r++)
        if (recvcounts[r]>0) {
            MPI_Request req;
            MPI_Irecv(recvbuf+rdispls[r],recvcounts[r],MPI_DOUBLE,r,tag,comm,&req);
            ex.requests.push_back(req);
        }
    for (int r=0;r<nranks;r++)
        if (sendcounts[r]>0) {
            MPI_Request req;
            MPI_Isend(sendbuf+sdispls[r],sendcounts[r],MPI_DOUBLE,r,tag,comm,&req);
            ex.requests.push_back(req);
            ex.bytes+=(long long)sendcounts[r]*(long long)sizeof(double);
            ex.messages++;
        }
}

void wait_exchange_active_peers(PeerExchange& ex) {
    if (!ex.requests.empty())
        MPI_Waitall((int)ex.requests.size(),ex.requests.data(),MPI_STATUSES_IGNORE);
    ex.requests.clear();
}

/* Frozen Stage 2 behaviour: post and wait fused. */
void exchange_active_peers(const double* sendbuf,
                           const std::vector<int>& sendcounts,
                           const std::vector<int>& sdispls,
                           double* recvbuf,
                           const std::vector<int>& recvcounts,
                           const std::vector<int>& rdispls,
                           int tag, MPI_Comm comm,
                           long long* bytes_out=nullptr,
                           long long* messages_out=nullptr) {
    PeerExchange ex;
    post_exchange_active_peers(ex,sendbuf,sendcounts,sdispls,
                               recvbuf,recvcounts,rdispls,tag,comm);
    if (bytes_out) *bytes_out+=ex.bytes;
    if (messages_out) *messages_out+=ex.messages;
    wait_exchange_active_peers(ex);
}

void ensure_dlb_stream() {
    if (g_dlb_stream) return;
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_dlb_stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ev_inputs_ready,cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ev_dlb_done,cudaEventDisableTiming));
}

void rebuild_dlb_plan(DevFields& f, const Dims& d, double tol, MPI_Comm comm) {
    const double plan_start=MPI_Wtime();
    int rank=0, nranks=1;
    MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&nranks);

    // Check rank totals before gathering per-cell Nchem.
    long long local_load=launch_sum_nchem(f,d);
    std::vector<long long> rank_loads(nranks,0);
    MPI_Allgather(&local_load,1,MPI_LONG_LONG_INT,
                  rank_loads.data(),1,MPI_LONG_LONG_INT,comm);
    const long long load_hi=*std::max_element(rank_loads.begin(),rank_loads.end());
    const long long load_sum=std::accumulate(rank_loads.begin(),rank_loads.end(),0LL);
    const double load_avg=nranks?double(load_sum)/nranks:0.0;
    const double load_degree=load_hi?double(load_hi-load_avg)/load_hi:0.0;

    g_send_tasks.assign(nranks,0);
    g_recv_tasks.assign(nranks,0);
    g_out_cells.clear();
    CUDA_CHECK(cudaMemset(f.DLBmask,0,(size_t)d.Ni*d.Nj*d.Nk*sizeof(int)));
    if (load_degree<=tol) {
        g_plan_ready=true;
        g_plan_has_tasks=false;
        g_dlb_timing.replans++;
        g_dlb_timing.skipped_replans++;
        g_dlb_timing.plan+=MPI_Wtime()-plan_start;
        if (rank==0)
            fprintf(stdout,"GPU DLB skip: LoadDegree=%.6g <= tol=%g\n",load_degree,tol);
        return;
    }

    const int local_count=d.ni*d.nj*d.nk;
    ensure_device(g_d_compact,g_compact_cap,(size_t)local_count);
    int threads=256, blocks=(local_count+threads-1)/threads;
    kCompactNchem<<<blocks,threads>>>(f.Nchem,g_d_compact,d);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> local(local_count);
    CUDA_CHECK(cudaMemcpy(local.data(),g_d_compact,(size_t)local_count*sizeof(int),cudaMemcpyDeviceToHost));

    std::vector<int> counts(nranks),displs(nranks,0);
    MPI_Allgather(&local_count,1,MPI_INT,counts.data(),1,MPI_INT,comm);
    for (int r=1;r<nranks;r++) displs[r]=displs[r-1]+counts[r-1];
    std::vector<int> all;
    if (rank==0) all.resize(displs.back()+counts.back());
    MPI_Gatherv(local.data(),local_count,MPI_INT,
                rank==0?all.data():nullptr,counts.data(),displs.data(),MPI_INT,0,comm);

    std::vector<int> plan;
    long long gate_rejected=0, budget_stop=0;
    if (rank==0) {
        DlbGateConfig gate;
        gate.tolerance=tol;
        if (dlb_cost_gate_requested()) {
            gate.min_task_weight=dlb_min_task_weight();
            gate.max_tasks_per_replan=dlb_max_tasks_per_replan();
        }
        DlbScheduleResult schedule=build_dlb_schedule(all,counts,displs,gate);
        gate_rejected=schedule.rejected_by_weight_gate;
        budget_stop=schedule.stopped_on_budget?1:0;
        plan=std::move(schedule.plan);
        long long before_hi=*std::max_element(schedule.before.begin(),schedule.before.end());
        long long before_lo=*std::min_element(schedule.before.begin(),schedule.before.end());
        long long after_hi=*std::max_element(schedule.after.begin(),schedule.after.end());
        long long after_lo=*std::min_element(schedule.after.begin(),schedule.after.end());
        double before_degree=before_hi?double(before_hi-before_lo)/before_hi:0.0;
        double after_degree=after_hi?double(after_hi-after_lo)/after_hi:0.0;
        fprintf(stdout,"GPU DLB plan: tasks=%zu LoadDegree=%.6g -> %.6g load-range=%lld -> %lld\n",
                plan.size()/3,before_degree,after_degree,
                before_hi-before_lo,after_hi-after_lo);
    }

    int plan_ints=(int)plan.size();
    MPI_Bcast(&plan_ints,1,MPI_INT,0,comm);
    if (rank!=0) plan.resize(plan_ints);
    if (plan_ints) MPI_Bcast(plan.data(),plan_ints,MPI_INT,0,comm);

    // Keep the sender-local compact index for now: it is what indexes `local`,
    // the compacted Nchem array, and therefore the task weight.
    std::vector<std::vector<int>> per_dest(nranks);
    for (int p=0;p<plan_ints;p+=3)
        if (plan[p]==rank)
            per_dest[plan[p+1]].push_back(plan[p+2]);

    const bool order_by_nchem=dlb_task_order_nchem();
    for (int r=0;r<nranks;r++) {
        auto& dest=per_dest[r];
        if (order_by_nchem) {
            // kSolveReactionTasks maps one thread to one task, so a warp spans 32
            // consecutive records.  Ordering by descending Nchem groups tasks of
            // similar iteration count into the same warp; without it a warp can
            // mix an Nchem=1 task with an Nchem=275 one and stall on the latter.
            //
            // This gives up the address-ordered coalescing of pack/unpack, which
            // the instrumented run measured at pack_kernel=0.019 s and
            // unpack_kernel=0.016 s cumulative over the whole 2834-step case, so
            // the trade is worth making.  Ties break on the compact index to keep
            // the plan deterministic.
            std::sort(dest.begin(),dest.end(),[&](int a,int b) {
                if (local[a]!=local[b]) return local[a]>local[b];
                return a<b;
            });
        } else {
            // Frozen Stage 2 behaviour: the schedule is ordered by load, not by
            // memory address, so sorting the final per-peer list keeps pack and
            // unpack accesses substantially more coalesced without changing which
            // cells are migrated.  compact_to_cell is strictly increasing in the
            // compact index, so sorting the compact indices is identical to
            // sorting the cell addresses as Stage 2 did.
            std::sort(dest.begin(),dest.end());
        }
        g_send_tasks[r]=(int)dest.size();
        for (int compact : dest) g_out_cells.push_back(compact_to_cell(compact,d));
    }
    MPI_Alltoall(g_send_tasks.data(),1,MPI_INT,g_recv_tasks.data(),1,MPI_INT,comm);

    if (!g_out_cells.empty()) {
        ensure_device(g_d_cells,g_cells_cap,g_out_cells.size());
        CUDA_CHECK(cudaMemcpy(g_d_cells,g_out_cells.data(),g_out_cells.size()*sizeof(int),cudaMemcpyHostToDevice));
        int n=(int)g_out_cells.size();
        kSetDLBMask<<<(n+255)/256,256>>>(f.DLBmask,g_d_cells,n);
        CUDA_CHECK(cudaGetLastError());
    }
    g_plan_ready=true;
    g_plan_has_tasks=plan_ints>0;
    g_dlb_timing.replans++;
    g_dlb_timing.planned_tasks+=plan_ints/3;
    g_dlb_timing.gate_rejected+=gate_rejected;
    g_dlb_timing.budget_stops+=budget_stop;
    if (!g_plan_has_tasks) g_dlb_timing.skipped_replans++;
    g_dlb_timing.plan+=MPI_Wtime()-plan_start;
}

/*---------------------------------------------------------------------------------
 * Intra-GPU bucket scheduling: refresh the cell list and decide whether the
 * counting sort is worth running at all.
 *
 * Called on the same cadence as the DLB replan so no per-step host round trip is
 * needed.  Returns true when the sorted list is populated and should be used.
 *---------------------------------------------------------------------------------*/
void ensure_bucket_buffers(const Dims& d) {
    const size_t interior=(size_t)d.ni*d.nj*d.nk;
    ensure_device(g_d_cell_list,g_cell_list_cap,interior);
    if (g_bucket_ready) return;
    CUDA_CHECK(cudaMalloc(&g_d_bucket_counts,EU3D_NCHEM_BUCKETS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_bucket_offsets,(EU3D_NCHEM_BUCKETS+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_bucket_cursors,EU3D_NCHEM_BUCKETS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_d_queue_heads,EU3D_NCHEM_BUCKETS*sizeof(int)));
    g_bucket_ready=true;
}

/* Build the Nchem-bucketed cell list on `stream`.  `mask` may be null. */
void build_bucket_cell_list(const int* Nchem, const int* mask,
                            const Dims& d, cudaStream_t stream) {
    ensure_bucket_buffers(d);
    const int interior=d.ni*d.nj*d.nk;
    const int threads=256;
    const int blocks=(interior+threads-1)/threads;
    CUDA_CHECK(cudaMemsetAsync(g_d_bucket_counts,0,
                               EU3D_NCHEM_BUCKETS*sizeof(int),stream));
    kCountNchemBuckets<<<blocks,threads,0,stream>>>(Nchem,mask,g_d_bucket_counts,d);
    CUDA_CHECK(cudaGetLastError());
    kNchemBucketOffsets<<<1,1,0,stream>>>(g_d_bucket_counts,g_d_bucket_offsets,
                                          g_d_bucket_cursors);
    CUDA_CHECK(cudaGetLastError());
    kScatterNchemBuckets<<<blocks,threads,0,stream>>>(Nchem,mask,g_d_bucket_cursors,
                                                      g_d_cell_list,d);
    CUDA_CHECK(cudaGetLastError());
}

/* Decide, once per replan cadence, whether bucketing can help this rank.  A rank
 * whose cells are all Nchem<=1 has zero warp divergence to recover, so paying for
 * two extra passes over the interior would be a pure loss. */
void refresh_bucket_decision(const int* Nchem, const int* mask,
                             const Dims& d, cudaStream_t stream) {
    ensure_bucket_buffers(d);
    const int interior=d.ni*d.nj*d.nk;
    const int threads=256;
    const int blocks=(interior+threads-1)/threads;
    CUDA_CHECK(cudaMemsetAsync(g_d_bucket_counts,0,
                               EU3D_NCHEM_BUCKETS*sizeof(int),stream));
    kCountNchemBuckets<<<blocks,threads,0,stream>>>(Nchem,mask,g_d_bucket_counts,d);
    CUDA_CHECK(cudaGetLastError());
    int counts[EU3D_NCHEM_BUCKETS]={0};
    CUDA_CHECK(cudaMemcpyAsync(counts,g_d_bucket_counts,sizeof(counts),
                               cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    long long total=0, heavy=0;
    for (int b=0;b<EU3D_NCHEM_BUCKETS;b++) {
        total+=counts[b];
        if (b>0) heavy+=counts[b];
    }
    const double fraction=total?double(heavy)/double(total):0.0;
    g_bucket_useful=fraction>=reaction_sched_min_heavy_fraction();
    g_bucket_evaluations++;
    if (!g_bucket_useful) g_bucket_bypasses++;
}

/* Single entry point for the local (non-migrated) reaction so that both the
 * legacy and the overlap timelines pick up EU3D_REACTION_SCHED automatically.
 * `mask` may be null when no DLB plan is active. */
void launch_local_reaction(DevFields& f, const DevConst& c, double dtm,
                           const Dims& d, ReactConst rc, const int* mask,
                           cudaStream_t stream) {
    if (reaction_sched_sorted() && g_bucket_useful) {
        build_bucket_cell_list(f.Nchem,mask,d,stream);
        const int interior=d.ni*d.nj*d.nk;
        const int threads=256;
        const int blocks=(interior+threads-1)/threads;
        if (reaction_sched_dynamic()) {
            // Bound the persistent worker pool by this kernel's residency.
            // All queue initialisation and consumption use the caller's stream.
            if (!g_queue_blocks) {
                int device=0, sms=0, resident=0;
                CUDA_CHECK(cudaGetDevice(&device));
                CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &resident,kTrapezoidWarpQueue,threads,0));
                g_queue_blocks=std::max(1,sms*resident);
            }
            CUDA_CHECK(cudaMemsetAsync(g_d_queue_heads,0,
                                       EU3D_NCHEM_BUCKETS*sizeof(int),stream));
            kTrapezoidWarpQueue<<<std::min(blocks,g_queue_blocks),threads,0,stream>>>(
                f.Mc,f.Di,f.T,f.E,f.Nchem,g_d_cell_list,g_d_bucket_offsets,
                g_d_queue_heads,f.CMS,c.Ri,dtm,rc,d);
            g_queue_launches++;
        } else {
            kTrapezoidList<<<blocks,threads,0,stream>>>(
                f.Mc,f.Di,f.T,f.E,f.Nchem,g_d_cell_list,g_d_bucket_offsets,
                f.CMS,c.Ri,dtm,rc,d);
        }
    } else {
        dim3 b=rblk(), g=rgrdInner(d);
        kTrapezoid<<<g,b,0,stream>>>(f.Mc,f.Di,f.T,f.E,f.Nchem,mask,
                                     f.CMS,c.Ri,dtm,rc,d);
    }
    CUDA_CHECK(cudaGetLastError());
}

void execute_dlb_reaction(DevFields& f,const DevConst& c,double dtm,const Dims& d,
                          ReactConst rc,MPI_Comm comm) {
    int nranks=1; MPI_Comm_size(comm,&nranks);
    const int NS=d.NS,in_stride=2*NS+3,out_stride=2*NS+1;
    int nsend=(int)g_out_cells.size();
    int nrecv=std::accumulate(g_recv_tasks.begin(),g_recv_tasks.end(),0);
    const bool cuda_aware=dlb_cuda_aware_requested();

    std::vector<int> send_in(nranks),recv_in(nranks),sd_in(nranks,0),rd_in(nranks,0);
    std::vector<int> send_out(nranks),recv_out(nranks),sd_out(nranks,0),rd_out(nranks,0);
    for (int r=0;r<nranks;r++) {
        send_in[r]=g_send_tasks[r]*in_stride; recv_in[r]=g_recv_tasks[r]*in_stride;
        send_out[r]=g_recv_tasks[r]*out_stride; recv_out[r]=g_send_tasks[r]*out_stride;
        if (r) {
            sd_in[r]=sd_in[r-1]+send_in[r-1]; rd_in[r]=rd_in[r-1]+recv_in[r-1];
            sd_out[r]=sd_out[r-1]+send_out[r-1]; rd_out[r]=rd_out[r-1]+recv_out[r-1];
        }
    }

    ensure_device(g_d_task_in,g_task_in_cap,(size_t)std::max(nsend,nrecv)*in_stride);
    if (cuda_aware)
        ensure_device(g_d_task_recv,g_task_recv_cap,(size_t)nrecv*in_stride);
    ensure_device(g_d_task_out,g_task_out_cap,(size_t)nrecv*out_stride);
    ensure_device(g_d_task_return,g_task_return_cap,(size_t)nsend*out_stride);
    size_t n_send_in=(size_t)nsend*in_stride,n_recv_in=(size_t)nrecv*in_stride;
    size_t n_send_out=(size_t)nrecv*out_stride,n_recv_out=(size_t)nsend*out_stride;
    if (!cuda_aware) {
        ensure_pinned(g_h_task_send,g_h_send_cap,n_send_in);
        ensure_pinned(g_h_task_recv,g_h_recv_cap,n_recv_in);
        ensure_pinned(g_h_task_done,g_h_done_cap,n_send_out);
        ensure_pinned(g_h_task_return,g_h_return_cap,n_recv_out);
    }

    const bool probe=dlb_probe_requested();

    if (!dlb_overlap_requested()) {
        /*----------------------------------------------------------------------
         * Frozen Stage 2 timeline, kept verbatim so it stays the A/B reference.
         * The only additions are traffic counters and the optional probe barrier.
         *--------------------------------------------------------------------*/
        double phase=MPI_Wtime();
        if (nsend) {
            kPackReactionTasks<<<(nsend+255)/256,256>>>(g_d_cells,nsend,f.Nchem,f.T,f.E,f.Di,f.Mc,
                                                        g_d_task_in,NS);
            CUDA_CHECK(cudaGetLastError());
            if (cuda_aware)
                CUDA_CHECK(cudaDeviceSynchronize());
            else
                CUDA_CHECK(cudaMemcpy(g_h_task_send,g_d_task_in,n_send_in*sizeof(double),cudaMemcpyDeviceToHost));
        }
        g_dlb_timing.pack+=MPI_Wtime()-phase;

        phase=MPI_Wtime();
        if (probe) {
            double skew0=MPI_Wtime();
            MPI_Barrier(comm);
            g_dlb_timing.forward_skew+=MPI_Wtime()-skew0;
        }
        double transfer0=MPI_Wtime();
        launch_local_reaction(f,c,dtm,d,rc,f.DLBmask,0);
        if (cuda_aware)
            exchange_active_peers(g_d_task_in,send_in,sd_in,
                                  g_d_task_recv,recv_in,rd_in,4201,comm,
                                  &g_dlb_timing.forward_bytes,
                                  &g_dlb_timing.forward_messages);
        else
            exchange_active_peers(g_h_task_send,send_in,sd_in,
                                  g_h_task_recv,recv_in,rd_in,4201,comm,
                                  &g_dlb_timing.forward_bytes,
                                  &g_dlb_timing.forward_messages);
        g_dlb_timing.forward_transfer+=MPI_Wtime()-transfer0;
        g_dlb_timing.forward+=MPI_Wtime()-phase;

        phase=MPI_Wtime();
        if (nrecv) {
            const double* task_input=g_d_task_in;
            if (cuda_aware)
                task_input=g_d_task_recv;
            else
                CUDA_CHECK(cudaMemcpy(g_d_task_in,g_h_task_recv,n_recv_in*sizeof(double),cudaMemcpyHostToDevice));
            kSolveReactionTasks<<<(nrecv+255)/256,256>>>(task_input,g_d_task_out,nrecv,dtm,c.Ri,rc,NS);
            CUDA_CHECK(cudaGetLastError());
            if (cuda_aware)
                CUDA_CHECK(cudaDeviceSynchronize());
            else
                CUDA_CHECK(cudaMemcpy(g_h_task_done,g_d_task_out,n_send_out*sizeof(double),cudaMemcpyDeviceToHost));
        }
        g_dlb_timing.remote+=MPI_Wtime()-phase;

        phase=MPI_Wtime();
        if (probe) {
            double skew0=MPI_Wtime();
            MPI_Barrier(comm);
            g_dlb_timing.backward_skew+=MPI_Wtime()-skew0;
        }
        transfer0=MPI_Wtime();
        if (cuda_aware)
            exchange_active_peers(g_d_task_out,send_out,sd_out,
                                  g_d_task_return,recv_out,rd_out,4202,comm,
                                  &g_dlb_timing.backward_bytes,
                                  &g_dlb_timing.backward_messages);
        else
            exchange_active_peers(g_h_task_done,send_out,sd_out,
                                  g_h_task_return,recv_out,rd_out,4202,comm,
                                  &g_dlb_timing.backward_bytes,
                                  &g_dlb_timing.backward_messages);
        g_dlb_timing.backward_transfer+=MPI_Wtime()-transfer0;
        g_dlb_timing.backward+=MPI_Wtime()-phase;

        phase=MPI_Wtime();
        if (nsend) {
            if (!cuda_aware)
                CUDA_CHECK(cudaMemcpy(g_d_task_return,g_h_task_return,n_recv_out*sizeof(double),cudaMemcpyHostToDevice));
            kUnpackReactionTasks<<<(nsend+255)/256,256>>>(g_d_cells,nsend,g_d_task_return,
                                                          f.T,f.Di,f.CMS,NS);
            CUDA_CHECK(cudaGetLastError());
        }
        g_dlb_timing.unpack+=MPI_Wtime()-phase;
        g_dlb_timing.active_steps++;
        return;
    }

    /* Pack and exchange inputs, overlap remote and local reaction, then return
     * results. Local and migrated cells are disjoint. A completion event orders
     * the following explicit update after unpacking. */
    ensure_dlb_stream();
    // The DLB stream must observe the prediction results produced on stream 0.
    CUDA_CHECK(cudaEventRecord(g_ev_inputs_ready,0));
    CUDA_CHECK(cudaStreamWaitEvent(g_dlb_stream,g_ev_inputs_ready,0));

    double phase=MPI_Wtime();
    if (nsend) {
        kPackReactionTasks<<<(nsend+255)/256,256,0,g_dlb_stream>>>(
            g_d_cells,nsend,f.Nchem,f.T,f.E,f.Di,f.Mc,g_d_task_in,NS);
        CUDA_CHECK(cudaGetLastError());
        if (!cuda_aware)
            CUDA_CHECK(cudaMemcpyAsync(g_h_task_send,g_d_task_in,
                                       n_send_in*sizeof(double),
                                       cudaMemcpyDeviceToHost,g_dlb_stream));
        // Wait only for the stream that produces the send buffer.
        CUDA_CHECK(cudaStreamSynchronize(g_dlb_stream));
    }
    g_dlb_timing.pack+=MPI_Wtime()-phase;

    phase=MPI_Wtime();
    if (probe) {
        double skew0=MPI_Wtime();
        MPI_Barrier(comm);
        g_dlb_timing.forward_skew+=MPI_Wtime()-skew0;
    }
    double transfer0=MPI_Wtime();
    PeerExchange fwd;
    post_exchange_active_peers(fwd,
                               cuda_aware?g_d_task_in:g_h_task_send,send_in,sd_in,
                               cuda_aware?g_d_task_recv:g_h_task_recv,recv_in,rd_in,
                               4201,comm);
    g_dlb_timing.forward_bytes+=fwd.bytes;
    g_dlb_timing.forward_messages+=fwd.messages;
    wait_exchange_active_peers(fwd);
    g_dlb_timing.forward_transfer+=MPI_Wtime()-transfer0;
    g_dlb_timing.forward+=MPI_Wtime()-phase;

    phase=MPI_Wtime();
    if (nrecv) {
        const double* task_input=g_d_task_in;
        if (cuda_aware) {
            task_input=g_d_task_recv;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(g_d_task_in,g_h_task_recv,
                                       n_recv_in*sizeof(double),
                                       cudaMemcpyHostToDevice,g_dlb_stream));
        }
        kSolveReactionTasks<<<(nrecv+255)/256,256,0,g_dlb_stream>>>(
            task_input,g_d_task_out,nrecv,dtm,c.Ri,rc,NS);
        CUDA_CHECK(cudaGetLastError());
        if (!cuda_aware)
            CUDA_CHECK(cudaMemcpyAsync(g_h_task_done,g_d_task_out,
                                       n_send_out*sizeof(double),
                                       cudaMemcpyDeviceToHost,g_dlb_stream));
    }
    // Launch local work now so it runs beside the migrated solve and covers the
    // backward exchange.  Stream 0 keeps the ordering with the rest of the step.
    launch_local_reaction(f,c,dtm,d,rc,f.DLBmask,0);
    if (nrecv) {
        // Waits for the migrated solve only; the non-blocking DLB stream does not
        // synchronise with stream 0, so the local kernel keeps running.
        CUDA_CHECK(cudaStreamSynchronize(g_dlb_stream));
    }
    g_dlb_timing.remote+=MPI_Wtime()-phase;

    phase=MPI_Wtime();
    if (probe) {
        double skew0=MPI_Wtime();
        MPI_Barrier(comm);
        g_dlb_timing.backward_skew+=MPI_Wtime()-skew0;
    }
    transfer0=MPI_Wtime();
    PeerExchange bwd;
    post_exchange_active_peers(bwd,
                               cuda_aware?g_d_task_out:g_h_task_done,send_out,sd_out,
                               cuda_aware?g_d_task_return:g_h_task_return,recv_out,rd_out,
                               4202,comm);
    g_dlb_timing.backward_bytes+=bwd.bytes;
    g_dlb_timing.backward_messages+=bwd.messages;
    wait_exchange_active_peers(bwd);
    g_dlb_timing.backward_transfer+=MPI_Wtime()-transfer0;
    g_dlb_timing.backward+=MPI_Wtime()-phase;

    phase=MPI_Wtime();
    if (nsend) {
        if (!cuda_aware)
            CUDA_CHECK(cudaMemcpyAsync(g_d_task_return,g_h_task_return,
                                       n_recv_out*sizeof(double),
                                       cudaMemcpyHostToDevice,g_dlb_stream));
        kUnpackReactionTasks<<<(nsend+255)/256,256,0,g_dlb_stream>>>(
            g_d_cells,nsend,g_d_task_return,f.T,f.Di,f.CMS,NS);
        CUDA_CHECK(cudaGetLastError());
    }
    // kExplicit / UpdateAfterCS run on stream 0 and read T/Di/CMS, so stream 0
    // must not overtake the unpack.
    CUDA_CHECK(cudaEventRecord(g_ev_dlb_done,g_dlb_stream));
    CUDA_CHECK(cudaStreamWaitEvent(0,g_ev_dlb_done,0));
    g_dlb_timing.unpack+=MPI_Wtime()-phase;
    g_dlb_timing.active_steps++;
}
} // namespace

void launch_update_after_adv(DevFields& f, const DevConst& c, const Dims& d) {
    dim3 b=rblk(), g=rgrdInner(d);
    kUpdateAfterAdv<<<g,b>>>(f.CS, f.RHS, d);
    kUpdateAfterCS<<<g,b>>>(f.CS,f.D,f.U,f.V,f.W,f.E,f.P,f.H,f.T,f.Gamma,f.C,f.Ma,
                            f.Yi,f.Di,f.Mc,c.Mw,c.Ri,c.Coeff0,c.Coeff1,c.R,d);
    CUDA_CHECK(cudaGetLastError());
}

void launch_trapezoid_prediction(DevFields& f, const DevConst& c, double dtm, const Dims& d) {
    dim3 b=rblk(), g=rgrdInner(d);
    ReactConst rc=makeRC(c);
    kTrapezoidPrediction<<<g,b>>>(f.Mc,f.Di,f.Yi,f.T,f.Nchem,dtm,rc,d);
    CUDA_CHECK(cudaGetLastError());
}

void launch_trapezoid(DevFields& f, const DevConst& c, double dtm, const Dims& d,
                      int iteration, int dlb_step, double dlb_tol, MPI_Comm comm) {
    ReactConst rc=makeRC(c);
    int nranks=1; MPI_Comm_size(comm,&nranks);
    const bool dlb_active=(nranks>1 && dlb_step>0);

    if (dlb_active && (!g_plan_ready || iteration%dlb_step==0))
        rebuild_dlb_plan(f,d,dlb_tol,comm);

    // Re-evaluate whether the Nchem counting sort pays off on this rank.  Doing it
    // on the replan cadence keeps the per-step path free of host round trips, and
    // the Nchem distribution only drifts as fast as the detonation front moves.
    if (reaction_sched_sorted()) {
        const int cadence=dlb_step>0?dlb_step:20;
        if (g_bucket_evaluations==0 || iteration%cadence==0)
            refresh_bucket_decision(f.Nchem,dlb_active?f.DLBmask:nullptr,d,0);
    }

    if (dlb_active && g_plan_has_tasks)
        execute_dlb_reaction(f,c,dtm,d,rc,comm);
    else
        launch_local_reaction(f,c,dtm,d,rc,nullptr,0);
    CUDA_CHECK(cudaGetLastError());
}

void launch_explicit(DevFields& f, const DevConst& c, const Dims& d) {
    dim3 b=rblk(), g=rgrdInner(d);
    kExplicit<<<g,b>>>(f.CS,f.Di,f.D,f.U,f.V,f.W,f.E,d);
    kUpdateAfterCS<<<g,b>>>(f.CS,f.D,f.U,f.V,f.W,f.E,f.P,f.H,f.T,f.Gamma,f.C,f.Ma,
                            f.Yi,f.Di,f.Mc,c.Mw,c.Ri,c.Coeff0,c.Coeff1,c.R,d);
    CUDA_CHECK(cudaGetLastError());
}

/* Sum interior Nchem using atomic reduction; exclude ghost cells. */
__global__ void kSumNchemInterior(const int* __restrict__ Nchem, unsigned long long* __restrict__ out, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    atomicAdd(out, (unsigned long long)Nchem[idx3(i,j,k,d.Nj,d.Nk)]);
}

__global__ void kMaxNchemInterior(const int* __restrict__ Nchem, int* __restrict__ out, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x + d.bc;
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    atomicMax(out, Nchem[idx3(i,j,k,d.Nj,d.Nk)]);
}

// Return the local interior Nchem sum.
static unsigned long long* g_sum_out = nullptr;
static int* g_max_out = nullptr;

long long launch_sum_nchem(DevFields& f, const Dims& d) {
    unsigned long long h_out=0;
    if (!g_sum_out) CUDA_CHECK(cudaMalloc(&g_sum_out, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_sum_out, 0, sizeof(unsigned long long)));
    dim3 b=rblk(), g=rgrdInner(d);
    kSumNchemInterior<<<g,b>>>(f.Nchem, g_sum_out, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_out, g_sum_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return (long long)h_out;
}

int launch_max_nchem(DevFields& f, const Dims& d) {
    int h_out=0;
    if (!g_max_out) CUDA_CHECK(cudaMalloc(&g_max_out, sizeof(int)));
    CUDA_CHECK(cudaMemset(g_max_out, 0, sizeof(int)));
    dim3 b=rblk(), g=rgrdInner(d);
    kMaxNchemInterior<<<g,b>>>(f.Nchem, g_max_out, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_out, g_max_out, sizeof(int), cudaMemcpyDeviceToHost));
    return h_out;
}

void finalize_reaction_reduction() {
    int mpi_initialized=0, mpi_finalized=0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) MPI_Finalized(&mpi_finalized);
    if (mpi_initialized && !mpi_finalized) {
        double local_times[10]={g_dlb_timing.plan,g_dlb_timing.pack,g_dlb_timing.forward,
                                g_dlb_timing.remote,g_dlb_timing.backward,g_dlb_timing.unpack,
                                g_dlb_timing.forward_skew,g_dlb_timing.forward_transfer,
                                g_dlb_timing.backward_skew,g_dlb_timing.backward_transfer};
        double max_times[10]={0,0,0,0,0,0,0,0,0,0};
        long long local_counts[8]={g_dlb_timing.replans,g_dlb_timing.skipped_replans,
                                   g_dlb_timing.active_steps,g_dlb_timing.planned_tasks,
                                   g_dlb_timing.gate_rejected,g_dlb_timing.budget_stops,
                                   g_bucket_evaluations,g_bucket_bypasses};
        long long max_counts[8]={0,0,0,0,0,0,0,0};
        long long local_traffic[4]={g_dlb_timing.forward_bytes,g_dlb_timing.backward_bytes,
                                    g_dlb_timing.forward_messages,g_dlb_timing.backward_messages};
        long long sum_traffic[4]={0,0,0,0};
        MPI_Reduce(local_times,max_times,10,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(local_counts,max_counts,8,MPI_LONG_LONG_INT,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(local_traffic,sum_traffic,4,MPI_LONG_LONG_INT,MPI_SUM,0,MPI_COMM_WORLD);
        int rank=0;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        fprintf(stdout,"GPU reaction scheduler rank=%d mode=%s dynamic_launches=%lld "
                       "bucket_evaluations=%lld bucket_bypasses=%lld worker_blocks=%d chunk=32\n",
                rank,reaction_sched_dynamic()?"dynamic":(reaction_sched_sorted()?"sorted":"baseline"),
                g_queue_launches,g_bucket_evaluations,g_bucket_bypasses,g_queue_blocks);
        if (rank==0 && max_counts[0]>0) {
            fprintf(stdout,"GPU DLB transport: %s\n",
                    dlb_cuda_aware_requested()?"cuda-aware-device":"pinned-host-staging");
            fprintf(stdout,"GPU DLB config: schedule=%s task_order=%s reaction_sched=%s "
                    "cost_gate=%s min_task_weight=%d max_tasks=%lld probe=%s\n",
                    dlb_overlap_requested()?"overlap":"serial",
                    dlb_task_order_nchem()?"nchem":"address",
                    reaction_sched_dynamic()?"dynamic":(reaction_sched_sorted()?"sorted":"baseline"),
                    dlb_cost_gate_requested()?"on":"off",
                    dlb_cost_gate_requested()?dlb_min_task_weight():1,
                    dlb_cost_gate_requested()?dlb_max_tasks_per_replan():0LL,
                    dlb_probe_requested()?"on":"off");
            fprintf(stdout,
                    "GPU DLB timing(max-rank cumulative s): plan=%.6f pack=%.6f "
                    "forward=%.6f remote=%.6f backward=%.6f unpack=%.6f\n",
                    max_times[0],max_times[1],max_times[2],
                    max_times[3],max_times[4],max_times[5]);
            // forward/backward_skew are only non-zero under EU3D_DLB_PROBE=1.  The
            // split answers directly whether the recorded MPI cost is data movement
            // or simply waiting for the slowest rank to arrive.
            fprintf(stdout,
                    "GPU DLB mpi split(max-rank cumulative s): forward_skew=%.6f "
                    "forward_transfer=%.6f backward_skew=%.6f backward_transfer=%.6f\n",
                    max_times[6],max_times[7],max_times[8],max_times[9]);
            fprintf(stdout,
                    "GPU DLB counters: replans=%lld skipped=%lld active_steps=%lld "
                    "planned_tasks=%lld gate_rejected=%lld budget_stops=%lld\n",
                    max_counts[0],max_counts[1],max_counts[2],max_counts[3],
                    max_counts[4],max_counts[5]);
            fprintf(stdout,
                    "GPU reaction sched counters: bucket_evaluations=%lld "
                    "bucket_bypasses=%lld\n",
                    max_counts[6],max_counts[7]);
            fprintf(stdout,
                    "GPU DLB traffic(global cumulative): forward_bytes=%lld "
                    "backward_bytes=%lld forward_messages=%lld backward_messages=%lld\n",
                    sum_traffic[0],sum_traffic[1],sum_traffic[2],sum_traffic[3]);
            const double fwd_gib=double(sum_traffic[0])/(1024.0*1024.0*1024.0);
            const double bwd_gib=double(sum_traffic[1])/(1024.0*1024.0*1024.0);
            const double fwd_msg=sum_traffic[2]?double(sum_traffic[0])/double(sum_traffic[2]):0.0;
            const double fwd_bw=max_times[2]>0.0?double(sum_traffic[0])/max_times[2]/1.0e6:0.0;
            fprintf(stdout,
                    "GPU DLB traffic derived: forward=%.4f GiB backward=%.4f GiB "
                    "mean_forward_message=%.1f B effective_forward_MBps=%.3f\n",
                    fwd_gib,bwd_gib,fwd_msg,fwd_bw);
        }
    }
    if (g_sum_out) CUDA_CHECK(cudaFree(g_sum_out));
    if (g_max_out) CUDA_CHECK(cudaFree(g_max_out));
    g_sum_out = nullptr;
    g_max_out = nullptr;
    if (g_d_compact) CUDA_CHECK(cudaFree(g_d_compact));
    if (g_d_cells) CUDA_CHECK(cudaFree(g_d_cells));
    if (g_d_task_in) CUDA_CHECK(cudaFree(g_d_task_in));
    if (g_d_task_recv) CUDA_CHECK(cudaFree(g_d_task_recv));
    if (g_d_task_out) CUDA_CHECK(cudaFree(g_d_task_out));
    if (g_d_task_return) CUDA_CHECK(cudaFree(g_d_task_return));
    if (g_h_task_send) CUDA_CHECK(cudaFreeHost(g_h_task_send));
    if (g_h_task_recv) CUDA_CHECK(cudaFreeHost(g_h_task_recv));
    if (g_h_task_done) CUDA_CHECK(cudaFreeHost(g_h_task_done));
    if (g_h_task_return) CUDA_CHECK(cudaFreeHost(g_h_task_return));
    if (g_d_queue_heads) CUDA_CHECK(cudaFree(g_d_queue_heads));
    g_d_queue_heads=nullptr; g_queue_blocks=0; g_queue_launches=0;
    if (g_d_bucket_counts) CUDA_CHECK(cudaFree(g_d_bucket_counts));
    if (g_d_bucket_offsets) CUDA_CHECK(cudaFree(g_d_bucket_offsets));
    if (g_d_bucket_cursors) CUDA_CHECK(cudaFree(g_d_bucket_cursors));
    if (g_d_cell_list) CUDA_CHECK(cudaFree(g_d_cell_list));
    if (g_ev_inputs_ready) CUDA_CHECK(cudaEventDestroy(g_ev_inputs_ready));
    if (g_ev_dlb_done) CUDA_CHECK(cudaEventDestroy(g_ev_dlb_done));
    if (g_dlb_stream) CUDA_CHECK(cudaStreamDestroy(g_dlb_stream));
    g_d_bucket_counts=g_d_bucket_offsets=g_d_bucket_cursors=g_d_cell_list=nullptr;
    g_cell_list_cap=0;g_bucket_ready=false;g_bucket_useful=true;
    g_bucket_evaluations=g_bucket_bypasses=0;
    g_ev_inputs_ready=nullptr;g_ev_dlb_done=nullptr;g_dlb_stream=nullptr;
    g_d_compact=nullptr;g_d_cells=nullptr;g_d_task_in=nullptr;g_d_task_recv=nullptr;
    g_d_task_out=nullptr;g_d_task_return=nullptr;
    g_h_task_send=g_h_task_recv=g_h_task_done=g_h_task_return=nullptr;
    g_compact_cap=g_cells_cap=g_task_in_cap=g_task_recv_cap=0;
    g_task_out_cap=g_task_return_cap=0;
    g_h_send_cap=g_h_recv_cap=g_h_done_cap=g_h_return_cap=0;
    g_out_cells.clear();g_send_tasks.clear();g_recv_tasks.clear();
    g_plan_ready=false;g_plan_has_tasks=false;g_dlb_timing=DlbTiming{};
}
