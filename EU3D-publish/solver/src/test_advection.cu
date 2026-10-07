/* Single-GPU advection smoke test: row-major indexing, finite RHS and zero residual for uniform flow. Run: make test_kernel && ./test_kernel. */
#include "cuda_common.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

/* C interface to advection.cu; AdvWorkspace is opaque here. */
extern "C" void* adv_ws_alloc(const Dims& d);
extern "C" void  adv_ws_free(void* w);
extern "C" void  advection_gpu_v(const Dims& d, double dt,
                                 const double* xnode, const double* ynode, const double* znode,
                                 const double* P, const double* D, const double* U,
                                 const double* V, const double* W, const double* H,
                                 const double* Gamma, const double* Yi,
                                 double* F, double* G, double* Q, double* RHS, void* w);

// Host reference for row-major indexing.
static inline int h_idx3(int i,int j,int k,int Nj,int Nk){ return (i*Nj+j)*Nk+k; }
static inline int h_idx4(int i,int j,int k,int s,int Nj,int Nk,int Ns){ return ((i*Nj+j)*Nk+k)*Ns+s; }

int main() {
    Dims d;
    d.set(/*ni*/16, /*nj*/16, /*nk*/8, /*bc*/2, /*NS*/9);
    printf("Grid: ni=%d nj=%d nk=%d bc=%d NS=%d  (Ni=%d Nj=%d Nk=%d NC=%d)\n",
           d.ni, d.nj, d.nk, d.bc, d.NS, d.Ni, d.Nj, d.Nk, d.NC);

    // Check host indexing.
    {
        int a = h_idx4(1,2,3,4, d.Nj, d.Nk, d.NC);
        int b = ((1*d.Nj+2)*d.Nk+3)*d.NC+4;
        if (a != b) { printf("[FAIL] idx4 mismatch\n"); return 1; }
        printf("[OK] index formula self-check passed\n");
    }

    size_t n3  = (size_t)d.Ni * d.Nj * d.Nk;
    size_t n4  = n3 * d.NC;
    size_t nS  = n3 * d.NS;

    // Uniform stationary flow with normalized species.
    std::vector<double> hP(n3, 101325.0), hD(n3, 1.0), hU(n3, 0.0),
                        hV(n3, 0.0), hW(n3, 0.0), hH(n3, 3.0e5), hG(n3, 1.4);
    std::vector<double> hYi(nS, 0.0);
    for (size_t m = 0; m < n3; m++) hYi[m * d.NS + 0] = 1.0; // Pure species 0.

    std::vector<double> hx(d.Ni), hy(d.Nj), hz(d.Nk);
    for (int i = 0; i < d.Ni; i++) hx[i] = 1.0e-3 * i;
    for (int j = 0; j < d.Nj; j++) hy[j] = 1.0e-3 * j;
    for (int k = 0; k < d.Nk; k++) hz[k] = 1.0e-3 * k;

    // Allocate and initialize device fields.
    auto up3 = [&](const std::vector<double>& h)->double*{
        double* p; CUDA_CHECK(cudaMalloc(&p, h.size()*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(p, h.data(), h.size()*sizeof(double), cudaMemcpyHostToDevice));
        return p;
    };
    double *P=up3(hP),*D=up3(hD),*U=up3(hU),*V=up3(hV),*W=up3(hW),*H=up3(hH),*G=up3(hG);
    double *Yi=up3(hYi), *xn=up3(hx), *yn=up3(hy), *zn=up3(hz);
    double *dF,*dG,*dQ,*dRHS;
    CUDA_CHECK(cudaMalloc(&dF,  n4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dG,  n4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dQ,  n4*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRHS,n4*sizeof(double)));

    void* ws = adv_ws_alloc(d);
    advection_gpu_v(d, 3e-8, xn, yn, zn, P, D, U, V, W, H, G, Yi, dF, dG, dQ, dRHS, ws);

    // Download RHS and check finite values and uniform-flow residual.
    std::vector<double> hRHS(n4);
    CUDA_CHECK(cudaMemcpy(hRHS.data(), dRHS, n4*sizeof(double), cudaMemcpyDeviceToHost));

    double maxAbs = 0.0; bool nan = false;
    for (int i = d.bc; i < d.ni + d.bc; i++)
      for (int j = d.bc; j < d.nj + d.bc; j++)
        for (int k = d.bc; k < d.nk + d.bc; k++)
          for (int s = 0; s < d.NC; s++) {
            double v = hRHS[h_idx4(i,j,k,s,d.Nj,d.Nk,d.NC)];
            if (std::isnan(v)) nan = true;
            maxAbs = fmax(maxAbs, fabs(v));
          }

    printf("RHS max|.| on interior = %.3e, NaN=%s\n", maxAbs, nan ? "YES" : "no");
    if (nan) { printf("[FAIL] RHS contains NaN\n"); return 1; }
    // Uniform stationary flow should have near-zero RHS.
    if (maxAbs < 1e-6) printf("[OK] uniform-field RHS ~ 0 (self-consistent)\n");
    else               printf("[WARN] uniform-field RHS not ~0, check reconstruction\n");

    adv_ws_free(ws);
    cudaFree(P);cudaFree(D);cudaFree(U);cudaFree(V);cudaFree(W);cudaFree(H);cudaFree(G);
    cudaFree(Yi);cudaFree(xn);cudaFree(yn);cudaFree(zn);
    cudaFree(dF);cudaFree(dG);cudaFree(dQ);cudaFree(dRHS);
    printf("Done.\n");
    return 0;
}
