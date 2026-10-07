/* Grid dimensions, row-major indexing and CUDA error checks. */
#ifndef EU3D_CUDA_COMMON_CUH
#define EU3D_CUDA_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* CUDA error checking. */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Local MPI subdomain: Ni=ni+2*bc, Nj=nj+2*bc, Nk=nk+2*bc. */
struct Dims {
    int ni, nj, nk;   // Interior dimensions.
    int bc;           // Ghost layers per side.
    int NS;           // Species count.
    int Ni, Nj, Nk;   // Dimensions including ghost cells.
    int NC;           // Conserved components: NS+4.

    void set(int ni_, int nj_, int nk_, int bc_, int NS_) {
        ni = ni_; nj = nj_; nk = nk_; bc = bc_; NS = NS_;
        Ni = ni + 2 * bc;
        Nj = nj + 2 * bc;
        Nk = nk + 2 * bc;
        NC = NS + 4;
    }
};

/* Row-major indexing: (i*Nj+j)*Nk+k; append the component dimension for 4D arrays. */
__host__ __device__ __forceinline__
int idx3(int i, int j, int k, int Nj, int Nk) {
    return (i * Nj + j) * Nk + k;
}

__host__ __device__ __forceinline__
int idx4(int i, int j, int k, int s, int Nj, int Nk, int Ns) {
    return ((i * Nj + j) * Nk + k) * Ns + s;
}

#endif // EU3D_CUDA_COMMON_CUH
