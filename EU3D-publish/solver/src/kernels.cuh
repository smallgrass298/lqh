/* Kernel launch interfaces and device field structures. */
#ifndef EU3D_KERNELS_CUH
#define EU3D_KERNELS_CUH

#include "cuda_common.cuh"
#include <mpi.h>

/* Row-major device fields with the same layout as CPU arrays. */
struct DevFields {
    // 3D fields: (Ni,Nj,Nk).
    double *U,*V,*W,*P,*D,*T,*H,*E,*C,*Ma,*Gamma,*Cp,*Wav,*Rgas;
    // 4D fields.
    double *Yi,*Mc,*Di;   // (.,NS)
    double *CS,*F,*G,*Q,*RHS; // (.,NC=NS+4)
    // Reaction fields.
    int    *Nchem;        // (Ni,Nj,Nk)
    int    *DLBmask;      // 1 = reaction task is executed by a remote rank
    double *CMS;          // (.,NS)
};

/* Device reaction and thermodynamic constants. */
struct DevConst {
    int NR;
    double R, Ru, P0, Rgas_const;      // R=8.31434
    double *Stoi_F,*Stoi_B,*React_TB;  // (NS,NR)
    double *Af,*Bf,*Eaf;               // (NR)
    double *Mw,*Ri;                    // (NS)
    double *Coeff0,*Coeff1;            // (9,NS)
    // Coordinates including ghost cells.
    double *xnode,*ynode,*znode;
};

/* MPI process topology supplied by the CPU framework. */
struct Topo {
    int myid_x,myid_y,myid_z;
    int m_block_x,m_block_y,m_block_z;
    int m_left,m_right,m_front,m_back,m_down,m_up;
};

/* Host launch interfaces. */

// Apply physical boundaries on boundary ranks.
void launch_boundary(DevFields& f, const DevConst& c, const Topo& t, const Dims& d);

// Advection update.
void launch_advection(DevFields& f, const DevConst& c, double dt, const Dims& d);
void finalize_advection();

// Reaction stages.
void launch_update_after_adv(DevFields& f, const DevConst& c, const Dims& d);   // CS+=RHS; UpdateAfterCS
void launch_trapezoid_prediction(DevFields& f, const DevConst& c, double dtm, const Dims& d);
void launch_trapezoid(DevFields& f, const DevConst& c, double dtm, const Dims& d,
                      int iteration, int dlb_step, double dlb_tol, MPI_Comm comm);
void launch_explicit(DevFields& f, const DevConst& c, const Dims& d);          // Explicit + UpdateAfterCS

// Halo exchange through pinned host buffers or CUDA-aware MPI.
void launch_exchange(DevFields& f, double* dsend, double* drecv,
                     const Topo& t, const Dims& d, MPI_Comm comm);
void finalize_exchange();

// Sum Nchem over local interior cells.
long long launch_sum_nchem(DevFields& f, const Dims& d);
int launch_max_nchem(DevFields& f, const Dims& d);
void finalize_reaction_reduction();

#endif // EU3D_KERNELS_CUH
