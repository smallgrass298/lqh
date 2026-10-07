/* NASA nine-coefficient Cp/H/S polynomials and Newton temperature iteration. Coefficients use row-major (9,NS) storage: Cp rows 0..6, H rows 0..7, S rows 0..6 and 8. */
#ifndef EU3D_THERMO_CUH
#define EU3D_THERMO_CUH

#include "cuda_common.cuh"
#include <cmath>

#define THERMO_NCOEF 9   // Coefficients per species.

__host__ __device__ __forceinline__
double coeffAt(const double* C, int row, int SP, int NS) {
    return C[row * NS + SP];   // Row-major (9,NS).
}

/* Constant-pressure heat capacity. */
__device__ __forceinline__
double d_GetCpi(double T, double R, int SP, const double* C0, const double* C1, int NS) {
    const double* C = (T < 1000.0) ? C0 : C1;
    double iT = 1.0 / T;
    return R * (coeffAt(C,0,SP,NS)*iT*iT + coeffAt(C,1,SP,NS)*iT + coeffAt(C,2,SP,NS)
              + coeffAt(C,3,SP,NS)*T + coeffAt(C,4,SP,NS)*T*T
              + coeffAt(C,5,SP,NS)*T*T*T + coeffAt(C,6,SP,NS)*T*T*T*T);
}

/* Species enthalpy. */
__device__ __forceinline__
double d_GetHi(double T, double R, int SP, const double* C0, const double* C1, int NS) {
    const double* C = (T < 1000.0) ? C0 : C1;
    double T2=T*T, T3=T2*T, T4=T3*T, T5=T4*T;
    return R * (-coeffAt(C,0,SP,NS)/T + coeffAt(C,1,SP,NS)*log(T) + coeffAt(C,2,SP,NS)*T
              + coeffAt(C,3,SP,NS)*T2/2.0 + coeffAt(C,4,SP,NS)*T3/3.0
              + coeffAt(C,5,SP,NS)*T4/4.0 + coeffAt(C,6,SP,NS)*T5/5.0
              + coeffAt(C,7,SP,NS));
}

/* Species entropy. */
__device__ __forceinline__
double d_GetSi(double T, double R, int SP, const double* C0, const double* C1, int NS) {
    const double* C = (T < 1000.0) ? C0 : C1;
    double iT = 1.0/T, T2=T*T, T3=T2*T, T4=T3*T;
    return R * (-coeffAt(C,0,SP,NS)*iT*iT/2.0 - coeffAt(C,1,SP,NS)*iT + coeffAt(C,2,SP,NS)*log(T)
              + coeffAt(C,3,SP,NS)*T + coeffAt(C,4,SP,NS)*T2/2.0
              + coeffAt(C,5,SP,NS)*T3/3.0 + coeffAt(C,6,SP,NS)*T4/4.0
              + coeffAt(C,8,SP,NS));
}

/* Recover temperature from Yi, Ri and internal energy E (J/kg). Stop at |dT|<1e-6 or after 10 iterations. */
__device__ __forceinline__
double d_GetTemp(double T, const double* Yi, const double* Ri, double E,
                 const double* C0, const double* C1, int NS) {
    double T0 = T, T_temp = T;
    for (int cnt = 0; cnt < 10; cnt++) {
        T0 = T_temp;
        double sumYH = 0.0, sumYCp = 0.0, sumYR = 0.0;
        for (int s = 0; s < NS; s++) {
            double Cpi = d_GetCpi(T0, Ri[s], s, C0, C1, NS);
            double Hi  = d_GetHi (T0, Ri[s], s, C0, C1, NS);
            sumYH  += Yi[s] * Hi;
            sumYCp += Yi[s] * Cpi;
            sumYR  += Yi[s] * Ri[s];
        }
        double temp1 = (sumYH - E * 1e-3) - sumYR * T0;
        double temp2 = sumYCp - sumYR;
        T_temp = T0 - temp1 / temp2;
        if (fabs(T_temp - T0) < 1e-6) break;
    }
    return T_temp;
}

#endif // EU3D_THERMO_CUH
