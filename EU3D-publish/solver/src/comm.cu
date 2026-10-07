/* Pack/exchange/unpack ghost cells along x/y/z using pinned host buffers or CUDA-aware MPI. Each direction has two rounds. Buffer order: D,U,V,W,P,T,H,Gamma,C, then NS species. CPU reverse y/z rounds swap P/T in both pack and unpack; this implementation uses a fixed order. */
#include "cuda_common.cuh"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct CommFields {
    double *D,*U,*V,*W,*P,*T,*H,*Gamma,*C;   // 3D
    double *Yi;                              // 4D (.,NS)
};

namespace {
double* g_hsend = nullptr;
double* g_hrecv = nullptr;
size_t g_hcapacity = 0;

bool cuda_aware_requested() {
    static const bool requested = []() {
        const char* value = std::getenv("EU3D_CUDA_AWARE_MPI");
        return value && std::strcmp(value, "1") == 0;
    }();
    return requested;
}

void ensure_host_capacity(size_t count) {
    if (g_hcapacity >= count) return;
    if (g_hsend) CUDA_CHECK(cudaFreeHost(g_hsend));
    if (g_hrecv) CUDA_CHECK(cudaFreeHost(g_hrecv));
    CUDA_CHECK(cudaMallocHost((void**)&g_hsend, count*sizeof(double)));
    CUDA_CHECK(cudaMallocHost((void**)&g_hrecv, count*sizeof(double)));
    g_hcapacity = count;
}

/* One nonblocking send/receive pair.  The safe default stages through pinned
 * host memory and therefore works with ordinary OpenMPI packages.  A cluster
 * with verified CUDA-aware MPI can set EU3D_CUDA_AWARE_MPI=1 to send device
 * pointers directly. */
void mpi_exchange(double* dsend, double* drecv, int total,
                  int send_rank, int recv_rank, int tag, MPI_Comm comm) {
    MPI_Request req[2];
    MPI_Status st[2];
    if (cuda_aware_requested()) {
        MPI_Isend(dsend, total, MPI_DOUBLE, send_rank, tag, comm, &req[0]);
        MPI_Irecv(drecv, total, MPI_DOUBLE, recv_rank, tag, comm, &req[1]);
        MPI_Waitall(2, req, st);
        return;
    }

    ensure_host_capacity((size_t)total);
    CUDA_CHECK(cudaMemcpy(g_hsend, dsend, (size_t)total*sizeof(double),
                          cudaMemcpyDeviceToHost));
    MPI_Isend(g_hsend, total, MPI_DOUBLE, send_rank, tag, comm, &req[0]);
    MPI_Irecv(g_hrecv, total, MPI_DOUBLE, recv_rank, tag, comm, &req[1]);
    MPI_Waitall(2, req, st);
    if (recv_rank != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(drecv, g_hrecv, (size_t)total*sizeof(double),
                              cudaMemcpyHostToDevice));
}
} // namespace

// Directions: 0=x, 1=y, 2=z.
// Pack a bc-thick slab with fixed coordinate base+sign*t.
// Match the CPU loop order.
__global__ void kPack(const CommFields f, double* buf, int dir, int base, int sign,
                      int size, Dims d) {
    const int tx = blockIdx.x*blockDim.x+threadIdx.x;
    const int ty = blockIdx.y*blockDim.y+threadIdx.y;
    const int tz = blockIdx.z*blockDim.z+threadIdx.z;
    int a,b,t;
    int i,j,k,idx;
    if (dir==0) {            // x: loop (t=ii[bc], a=j[Nj], b=k[Nk])
        t=tx; a=ty; b=tz;
        if (t>=d.bc||a>=d.Nj||b>=d.Nk) return;
        i=base+sign*t; j=a; k=b; idx=(t*d.Nj+a)*d.Nk+b;
    } else if (dir==1) {     // y: loop (a=i[Ni], t=jj[bc], b=k[Nk])
        a=tx; t=ty; b=tz;
        if (a>=d.Ni||t>=d.bc||b>=d.Nk) return;
        i=a; j=base+sign*t; k=b; idx=(a*d.bc+t)*d.Nk+b;
    } else {                 // z: loop (a=i[Ni], b=j[Nj], t=kk[bc])
        a=tx; b=ty; t=tz;
        if (a>=d.Ni||b>=d.Nj||t>=d.bc) return;
        i=a; j=b; k=base+sign*t; idx=(a*d.Nj+b)*d.bc+t;
    }
    int c=idx3(i,j,k,d.Nj,d.Nk);
    buf[idx]        = f.D[c];
    buf[idx+size]   = f.U[c];
    buf[idx+2*size] = f.V[c];
    buf[idx+3*size] = f.W[c];
    buf[idx+4*size] = f.P[c];
    buf[idx+5*size] = f.T[c];
    buf[idx+6*size] = f.H[c];
    buf[idx+7*size] = f.Gamma[c];
    buf[idx+8*size] = f.C[c];
    for (int s=0;s<d.NS;s++)
        buf[9*size+idx*d.NS+s] = f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)];
}

__global__ void kUnpack(CommFields f, const double* buf, int dir, int base, int sign,
                        int size, Dims d) {
    const int tx = blockIdx.x*blockDim.x+threadIdx.x;
    const int ty = blockIdx.y*blockDim.y+threadIdx.y;
    const int tz = blockIdx.z*blockDim.z+threadIdx.z;
    int a,b,t;
    int i,j,k,idx;
    if (dir==0) {
        t=tx; a=ty; b=tz;
        if (t>=d.bc||a>=d.Nj||b>=d.Nk) return;
        i=base+sign*t; j=a; k=b; idx=(t*d.Nj+a)*d.Nk+b;
    } else if (dir==1) {
        a=tx; t=ty; b=tz;
        if (a>=d.Ni||t>=d.bc||b>=d.Nk) return;
        i=a; j=base+sign*t; k=b; idx=(a*d.bc+t)*d.Nk+b;
    } else {
        a=tx; b=ty; t=tz;
        if (a>=d.Ni||b>=d.Nj||t>=d.bc) return;
        i=a; j=b; k=base+sign*t; idx=(a*d.Nj+b)*d.bc+t;
    }
    int c=idx3(i,j,k,d.Nj,d.Nk);
    f.D[c]=buf[idx];        f.U[c]=buf[idx+size];   f.V[c]=buf[idx+2*size];
    f.W[c]=buf[idx+3*size]; f.P[c]=buf[idx+4*size]; f.T[c]=buf[idx+5*size];
    f.H[c]=buf[idx+6*size]; f.Gamma[c]=buf[idx+7*size]; f.C[c]=buf[idx+8*size];
    for (int s=0;s<d.NS;s++)
        f.Yi[idx4(i,j,k,s,d.Nj,d.Nk,d.NS)] = buf[9*size+idx*d.NS+s];
}

// Directional launch geometry and buffer size.
static void dirLaunch(int dir, const Dims& d, dim3& b, dim3& g, int& size) {
    // Keep the halo thickness on the axis normal to the exchanged face:
    // x exchange -> thread x, y exchange -> thread y, z exchange -> thread z.
    // kPack/kUnpack decode (tx,ty,tz) with this same convention.
    if (dir==0)      { b=dim3(1,8,8); g=dim3(1,(d.Nj+7)/8,(d.Nk+7)/8); size=d.Nk*d.Nj*d.bc; }
    else if (dir==1) { b=dim3(8,1,8); g=dim3((d.Ni+7)/8,1,(d.Nk+7)/8); size=d.Nk*d.Ni*d.bc; }
    else             { b=dim3(8,8,1); g=dim3((d.Ni+7)/8,(d.Nj+7)/8,1); size=d.Nj*d.Ni*d.bc; }
    // Use bc threads along the slab thickness.
    if (dir==0) b.x=d.bc; else if (dir==1) b.y=d.bc; else b.z=d.bc;
}

/* Two-round exchange along one axis. n is the interior extent; neighborLo/Hi are adjacent ranks and myidLo/Hi indicate their presence. */
static void exchange_dir(int dir, CommFields f, double* dsend, double* drecv,
                         int n, int neighborLo, int neighborHi,
                         bool hasLo, bool hasHi, Dims d, MPI_Comm comm,
                         int tagA, int tagB) {
    dim3 b,g; int size;
    dirLaunch(dir,d,b,g,size);
    int total = 9*size + d.NS*size;
    // Send high, receive low: pack n-t, unpack bc-1-t.
    kPack<<<g,b>>>(f, dsend, dir, n, -1, size, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    mpi_exchange(dsend, drecv, total, neighborHi, neighborLo, tagA, comm);
    if (hasLo) { kUnpack<<<g,b>>>(f, drecv, dir, d.bc-1, -1, size, d); CUDA_CHECK(cudaDeviceSynchronize()); }

    // Send low, receive high: pack bc+1+t, unpack n+bc+t.
    kPack<<<g,b>>>(f, dsend, dir, d.bc+1, +1, size, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    mpi_exchange(dsend, drecv, total, neighborLo, neighborHi, tagB, comm);
    if (hasHi) { kUnpack<<<g,b>>>(f, drecv, dir, n+d.bc, +1, size, d); CUDA_CHECK(cudaDeviceSynchronize()); }
}

/* Host launcher. */
#include "kernels.cuh"

void launch_exchange(DevFields& f, double* dsend, double* drecv,
                     const Topo& t, const Dims& d, MPI_Comm comm) {
    CommFields cf;
    cf.D=f.D; cf.U=f.U; cf.V=f.V; cf.W=f.W; cf.P=f.P;
    cf.T=f.T; cf.H=f.H; cf.Gamma=f.Gamma; cf.C=f.C; cf.Yi=f.Yi;

    static bool announced = false;
    if (!announced && (t.m_block_x*t.m_block_y*t.m_block_z > 1)) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0)
            fprintf(stdout, "GPU halo MPI mode: %s\n",
                    cuda_aware_requested() ? "CUDA-aware direct" : "pinned-host staging");
        announced = true;
    }

    // x direction.
    if (t.m_block_x > 1)
        exchange_dir(0, cf, dsend, drecv, d.ni, t.m_left, t.m_right,
                     t.myid_x!=0, t.myid_x!=t.m_block_x-1, d, comm, 0, 100);
    // y direction: front is low, back is high.
    if (t.m_block_y > 1)
        exchange_dir(1, cf, dsend, drecv, d.nj, t.m_front, t.m_back,
                     t.myid_y!=0, t.myid_y!=t.m_block_y-1, d, comm, 200, 300);
    // z direction: down is low, up is high.
    if (t.m_block_z > 1)
        exchange_dir(2, cf, dsend, drecv, d.nk, t.m_down, t.m_up,
                     t.myid_z!=0, t.myid_z!=t.m_block_z-1, d, comm, 400, 500);
}

void finalize_exchange() {
    if (g_hsend) CUDA_CHECK(cudaFreeHost(g_hsend));
    if (g_hrecv) CUDA_CHECK(cudaFreeHost(g_hrecv));
    g_hsend = g_hrecv = nullptr;
    g_hcapacity = 0;
}
