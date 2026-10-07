/* RSBI boundary conditions: x-low freestream, y-low symmetry, remaining faces outflow. Reflect V at y-low and W at z faces. Recompute thermodynamics at y-low. Only boundary ranks fill ghost cells. */
#include "cuda_common.cuh"
#include "thermo.cuh"

/* Copy a 3D field from (si,sj,sk) to (i,j,k). */
#define COPY3(FIELD) FIELD[idx3(i,j,k,d.Nj,d.Nk)] = FIELD[idx3(si,sj,sk,d.Nj,d.Nk)]

struct FieldPtrs {
    double *U,*V,*W,*P,*D,*T,*H,*Gamma,*E,*C,*Ma,*Cp,*Wav,*Rgas;
    double *Yi,*Mc;   // 4D (.,NS)
};

/* x-low freestream mirror: xl=2*bc-1. */
__global__ void kBC_xL(FieldPtrs f, Dims d) {
    int i = threadIdx.x;                                   // [0,bc)
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;      // [bc,nj+bc)
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;      // [bc,nk+bc)
    if (i>=d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int si = (2*d.bc-1) - i, sj = j, sk = k;
    COPY3(f.U);COPY3(f.V);COPY3(f.W);COPY3(f.P);COPY3(f.D);
    COPY3(f.T);COPY3(f.H);COPY3(f.Gamma);COPY3(f.E);
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
}

/* x-high outflow mirror: xr=2*(ni+bc)-1. */
__global__ void kBC_xR(FieldPtrs f, Dims d) {
    int i = d.ni+d.bc + threadIdx.x;                       // [ni+bc, ni+2bc)
    int j = blockIdx.y*blockDim.y+threadIdx.y + d.bc;
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+2*d.bc || j>=d.nj+d.bc || k>=d.nk+d.bc) return;
    int si = (2*(d.ni+d.bc)-1) - i, sj = j, sk = k;
    COPY3(f.U);COPY3(f.V);COPY3(f.W);COPY3(f.P);COPY3(f.D);
    COPY3(f.T);COPY3(f.H);COPY3(f.Gamma);COPY3(f.E);
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
}

/* y-low symmetry: yf=2*bc-1; reflect V and update thermodynamics. */
__global__ void kBC_yF(FieldPtrs f, Dims d,
                       const double* Ri, const double* Mw,
                       const double* C0, const double* C1, double Rgas_const) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;             // [0, ni+2bc)
    int j = threadIdx.y;                                   // [0,bc)
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;      // [bc,nk+bc)
    if (i>=d.ni+2*d.bc || j>=d.bc || k>=d.nk+d.bc) return;
    int si=i, sj=(2*d.bc-1)-j, sk=k;
    int c = idx3(i,j,k,d.Nj,d.Nk), s0 = idx3(si,sj,sk,d.Nj,d.Nk);

    f.U[c] =  f.U[s0];
    f.V[c] = -f.V[s0];   // Reverse the velocity normal to the symmetry plane.
    f.W[c] =  f.W[s0];
    f.P[c] =  f.P[s0];
    f.T[c] =  f.T[s0];

    double Tl = f.T[c];
    double sumYR=0.0, sumYCp=0.0, sumYH=0.0, sumYE=0.0;
    for (int s=0;s<d.NS;s++) {
        double y = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = y;
        f.Mc[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Mc[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
        double Cpi = d_GetCpi(Tl, Ri[s], s, C0, C1, d.NS);
        double Hi  = d_GetHi (Tl, Ri[s], s, C0, C1, d.NS);
        double Ei  = Hi - Ri[s]*Tl;
        sumYR  += y*Ri[s];
        sumYCp += y*Cpi;
        sumYH  += y*Hi;
        sumYE  += y*Ei;
    }
    f.Rgas[c] = sumYR*1000.0;
    f.D[c]    = f.P[c]/(Tl*f.Rgas[c]);
    f.Cp[c]   = sumYCp*1000.0;
    f.H[c]    = sumYH*1000.0;
    f.E[c]    = sumYE*1000.0;
    f.Gamma[c]= f.Cp[c]/(f.Cp[c]-f.Rgas[c]);
}

/* y-high outflow mirror: yb=2*(nj+bc)-1. */
__global__ void kBC_yB(FieldPtrs f, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;             // [0, ni+2bc)
    int j = d.nj+d.bc + threadIdx.y;                       // [nj+bc, nj+2bc)
    int k = blockIdx.z*blockDim.z+threadIdx.z + d.bc;
    if (i>=d.ni+2*d.bc || j>=d.nj+2*d.bc || k>=d.nk+d.bc) return;
    int si=i, sj=(2*(d.nj+d.bc)-1)-j, sk=k;
    COPY3(f.U);COPY3(f.V);COPY3(f.W);COPY3(f.P);COPY3(f.D);
    COPY3(f.T);COPY3(f.H);COPY3(f.Gamma);COPY3(f.E);
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
}

/* z-low mirror: zd=2*bc-1; reflect W. */
__global__ void kBC_zD(FieldPtrs f, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;             // [0, ni+2bc)
    int j = blockIdx.y*blockDim.y+threadIdx.y;             // [0, nj+2bc)
    int k = threadIdx.z;                                   // [0,bc)
    if (i>=d.ni+2*d.bc || j>=d.nj+2*d.bc || k>=d.bc) return;
    int si=i, sj=j, sk=(2*d.bc-1)-k;
    int c=idx3(i,j,k,d.Nj,d.Nk), s0=idx3(si,sj,sk,d.Nj,d.Nk);
    f.U[c]=f.U[s0]; f.V[c]=f.V[s0]; f.W[c]=-f.W[s0]; f.P[c]=f.P[s0]; f.D[c]=f.D[s0];
    f.T[c]=f.T[s0]; f.H[c]=f.H[s0]; f.Gamma[c]=f.Gamma[s0]; f.E[c]=f.E[s0];
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
}

/* z-high mirror: zu=2*(nk+bc)-1; reflect W. */
__global__ void kBC_zU(FieldPtrs f, Dims d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    int k = d.nk+d.bc + threadIdx.z;                       // [nk+bc, nk+2bc)
    if (i>=d.ni+2*d.bc || j>=d.nj+2*d.bc || k>=d.nk+2*d.bc) return;
    int si=i, sj=j, sk=(2*(d.nk+d.bc)-1)-k;
    int c=idx3(i,j,k,d.Nj,d.Nk), s0=idx3(si,sj,sk,d.Nj,d.Nk);
    f.U[c]=f.U[s0]; f.V[c]=f.V[s0]; f.W[c]=-f.W[s0]; f.P[c]=f.P[s0]; f.D[c]=f.D[s0];
    f.T[c]=f.T[s0]; f.H[c]=f.H[s0]; f.Gamma[c]=f.Gamma[s0]; f.E[c]=f.E[s0];
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = f.Yi[idx4(si,sj,sk,s,d.Nj,d.Nk,d.NS)];
}

/* Host launcher. */
#include "kernels.cuh"

void launch_boundary(DevFields& f, const DevConst& c, const Topo& t, const Dims& d) {
    FieldPtrs fp;
    fp.U=f.U; fp.V=f.V; fp.W=f.W; fp.P=f.P; fp.D=f.D; fp.T=f.T; fp.H=f.H;
    fp.Gamma=f.Gamma; fp.E=f.E; fp.C=f.C; fp.Ma=f.Ma; fp.Cp=f.Cp;
    fp.Wav=f.Wav; fp.Rgas=f.Rgas; fp.Yi=f.Yi; fp.Mc=f.Mc;

    // x faces.
    {
        dim3 b(d.bc,8,8);
        dim3 g(1,(d.nj+b.y-1)/b.y,(d.nk+b.z-1)/b.z);
        if (t.myid_x==0)             kBC_xL<<<g,b>>>(fp,d);
        if (t.myid_x==t.m_block_x-1) kBC_xR<<<g,b>>>(fp,d);
    }
    // y faces.
    {
        dim3 b(8,d.bc,8);
        dim3 g((d.ni+2*d.bc+b.x-1)/b.x,1,(d.nk+b.z-1)/b.z);
        if (t.myid_y==0)             kBC_yF<<<g,b>>>(fp,d,c.Ri,c.Mw,c.Coeff0,c.Coeff1,c.R);
        if (t.myid_y==t.m_block_y-1) kBC_yB<<<g,b>>>(fp,d);
    }
    // z faces.
    {
        dim3 b(8,8,d.bc);
        dim3 g((d.ni+2*d.bc+b.x-1)/b.x,(d.nj+2*d.bc+b.y-1)/b.y,1);
        if (t.myid_z==0)             kBC_zD<<<g,b>>>(fp,d);
        if (t.myid_z==t.m_block_z-1) kBC_zU<<<g,b>>>(fp,d);
    }
    CUDA_CHECK(cudaGetLastError());
}
