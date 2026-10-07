/* C interface between the CPU framework and CUDA solver. */
#ifndef EU3D_GPU_SOLVER_H
#define EU3D_GPU_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

void gpu_solver_init(
    int ni,int nj,int nk,int bc,int NS,int NR,
    const double* U,const double* V,const double* W,const double* P,const double* Dd,
    const double* Tt,const double* H,const double* E,const double* Cc,const double* Ma,
    const double* Gamma,const double* Cp,const double* Wav,const double* Rgas,
    const double* Yi,const double* Mc,const double* Di,const double* CS,
    const double* Stoi_F,const double* Stoi_B,const double* React_TB,
    const double* Af,const double* Bf,const double* Eaf,
    const double* Mw,const double* Ri,const double* Coeff0,const double* Coeff1,
    const double* xnode,const double* ynode,const double* znode,
    double Rconst,double Ru,double P0,
    int myid_x,int myid_y,int myid_z,int mbx,int mby,int mbz,
    int m_left,int m_right,int m_front,int m_back,int m_down,int m_up,
    int dlb_step,double dlb_tol);

void gpu_solver_step(double dt, int iteration);

/* Wait for all queued GPU work.  The single-rank fast path defers this
 * synchronization until timing/output boundaries instead of doing it every step. */
void gpu_solver_synchronize(void);

/* Sum Nchem over local interior cells. */
long long gpu_solver_sum_nchem(void);
int gpu_solver_max_nchem(void);

void gpu_solver_download(
    double* U,double* V,double* W,double* P,double* Dd,double* Tt,
    double* Yi,double* Gamma,double* Ma);

void gpu_solver_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
