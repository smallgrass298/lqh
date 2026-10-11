#!/usr/bin/env python3
"""Compare CPU/GPU Tecplot block outputs, one Ni*Nj*Nk block per variable."""
import argparse
import sys
import numpy as np

VAR_NAMES = ["X","Y","Z","T","D","P","U","V","Ma","Y1","Y2","Y3","Y4","Y5","Y6","Y7"]

def load(path):
    with open(path) as f:
        header1 = f.readline()
        header2 = f.readline()   # ZONE I=.. J=.. K=..
        f.readline()             # datapacking=block
        data = np.loadtxt(f)
    # Parse grid dimensions.
    parts = header2.replace("=", " ").split()
    I = int(parts[parts.index("I")+1])
    J = int(parts[parts.index("J")+1])
    K = int(parts[parts.index("K")+1])
    n = I*J*K
    nvar = len(data)//n
    assert nvar == len(VAR_NAMES), f"Variable block count mismatch: {nvar} vs {len(VAR_NAMES)}"
    # Output() writes values with i as the innermost loop, then j, then k.
    # Keep that geometry so diagnostics can distinguish interior cells from ghosts.
    return data.reshape(nvar, K, J, I), (I,J,K)


def compare_region(cpu, gpu, label, rel_l2_tol, abs_tol, offset=(0, 0, 0)):
    print(f"\n--- {label} ---")
    print(f"{'Var':<6}{'MaxAbsErr':<15}{'RelL2':<13}{'MaxRel*':<13}{'ScaleRef':<15}{'MaxAt(i,j,k)'}")
    print("MaxRel is diagnostic; acceptance uses RelL2 or absolute error.")
    all_ok = True
    oi, oj, ok = offset
    for i, name in enumerate(VAR_NAMES):
        a, b = cpu[i], gpu[i]
        abs_err = np.abs(a-b)
        scale = np.sqrt(np.mean(a*a))
        mask = np.abs(a) > max(scale, 1e-30) * 1e-6
        rel = (abs_err[mask]/np.abs(a[mask])).max() if mask.any() else 0.0
        denom = np.linalg.norm(a.ravel())
        rel_l2 = np.linalg.norm((a-b).ravel()) / max(denom, 1e-300)
        max_abs = abs_err.max()
        kz, jy, ix = np.unravel_index(np.argmax(abs_err), abs_err.shape)
        where = (int(ix + oi), int(jy + oj), int(kz + ok))
        # Near-zero fields are judged by an absolute floor; otherwise use a
        # global L2 norm, which is stable for shocks and reaction fronts.
        effective_abs_tol = max(abs_tol, max(scale, 1e-30)*1e-8)
        ok_var = (rel_l2 < rel_l2_tol) or (max_abs < effective_abs_tol)
        all_ok &= ok_var
        flag = "OK" if ok_var else "FAIL"
        print(f"{name:<6}{max_abs:<15.3e}{rel_l2:<13.3e}{rel:<13.3e}{scale:<15.3e}{str(where):<18} [{flag}]")
    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Compare EU3D CPU/GPU output fields")
    parser.add_argument("cpu_file")
    parser.add_argument("gpu_file")
    parser.add_argument("--bc", type=int, default=0,
                        help="Ghost layers to exclude for the interior comparison")
    parser.add_argument("--rel-l2-tol", type=float, default=1e-3,
                        help="Relative L2 tolerance (default: 1e-3)")
    parser.add_argument("--abs-tol", type=float, default=1e-6,
                        help="Absolute tolerance for near-zero fields (default: 1e-6)")
    args = parser.parse_args()
    cpu_path, gpu_path = args.cpu_file, args.gpu_file
    cpu, dims_c = load(cpu_path)
    gpu, dims_g = load(gpu_path)
    print(f"CPU dims={dims_c}  GPU dims={dims_g}")
    if dims_c != dims_g:
        print("[FAIL] Grid dimensions differ"); sys.exit(1)

    all_ok = compare_region(cpu, gpu, "Full domain including ghost cells",
                            args.rel_l2_tol, args.abs_tol)

    if args.bc:
        I, J, K = dims_c
        bc = args.bc
        if min(I, J, K) <= 2*bc:
            print(f"[FAIL] bc={bc} is incompatible with dimensions {dims_c}")
            sys.exit(1)
        interior_cpu = cpu[:, bc:K-bc, bc:J-bc, bc:I-bc]
        interior_gpu = gpu[:, bc:K-bc, bc:J-bc, bc:I-bc]
        all_ok &= compare_region(interior_cpu, interior_gpu,
                                 f"Interior excluding {bc} ghost layers",
                                 args.rel_l2_tol, args.abs_tol,
                                 offset=(bc, bc, bc))

    print("\n=== Result ===")
    if all_ok:
        print("[PASS] All compared fields satisfy the specified tolerances.")
    else:
        print("[FAIL] One or more fields exceed the tolerances")
        sys.exit(1)

if __name__ == "__main__":
    main()
