import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from decoherence_utils import *

N = 5


def get_all_velocities(rand_coeff, N=N):
    """
    Run N realisations and return the *raw* velocity components.
    """
    vx_all = []
    vy_all = []
    vx_cc_tde_all = []
    vy_cc_tde_all = []
    confidence_all = []
    vx_2dca_tde_all = []
    vy_2dca_tde_all = []
    cond_repr_all = []

    for _ in range(N):
        ds = make_decoherence_realization(rand_coeff)
        ds = im.run_norm_ds(ds, method_parameters["preprocessing"]["radius"])

        (
            v_c,
            w_c,
            vx_cc_tde,
            vy_cc_tde,
            confidence,
            vx_2dca_tde,
            vy_2dca_tde,
            cond_repr,
        ) = estimate_velocities(ds, method_parameters)

        vx_all.append(v_c)
        vy_all.append(w_c)
        vx_cc_tde_all.append(vx_cc_tde)
        vy_cc_tde_all.append(vy_cc_tde)
        confidence_all.append(confidence)
        vx_2dca_tde_all.append(vx_2dca_tde)
        vy_2dca_tde_all.append(vy_2dca_tde)
        cond_repr_all.append(cond_repr)

    return (
        vx_all,
        vy_all,
        vx_cc_tde_all,
        vy_cc_tde_all,
        confidence_all,
        vx_2dca_tde_all,
        vy_2dca_tde_all,
        cond_repr_all,
    )


# ----------------------------------------------------------------------
# Worker that includes index for ordering
# ----------------------------------------------------------------------
def _worker_with_index(args):
    idx, rand_coeff, N = args
    result = get_all_velocities(rand_coeff, N=N)
    return idx, result


# ----------------------------------------------------------------------
# Parallel driver using multiprocessing.Pool
# ----------------------------------------------------------------------
def parallel_loop_mp(rand_coeffs: List[float], N: int, processes: int = None):
    # Pre-allocate result containers
    n = len(rand_coeffs)
    vx_all = [None] * n
    vy_all = [None] * n
    vxtde_all = [None] * n
    vytde_all = [None] * n
    confidences = [None] * n
    vx_2dcas = [None] * n
    vy_2dcas = [None] * n
    cond_reprs = [None] * n

    # Prepare arguments: (index, coeff, N)
    args_list = [(i, rand_coeffs[i], N) for i in range(n)]

    with Pool(processes=processes) as pool:
        # imap_unordered yields results as they complete
        for idx, (
            vx,
            vy,
            vx_cc_tde_all,
            vy_cc_tde_all,
            confidence_all,
            vx_2dca_tde_all,
            vy_2dca_tde_all,
            cond_repr_all,
        ) in pool.imap_unordered(_worker_with_index, args_list):
            # Store in correct position
            vx_all[idx] = vx
            vy_all[idx] = vy
            vxtde_all[idx] = vx_cc_tde_all
            vytde_all[idx] = vy_cc_tde_all
            confidences[idx] = confidence_all
            vx_2dcas[idx] = vx_2dca_tde_all
            vy_2dcas[idx] = vy_2dca_tde_all
            cond_reprs[idx] = cond_repr_all

            print(f"Finished rand coeff = {rand_coeffs[idx]:.3f}")

    return (
        vx_all,
        vy_all,
        vxtde_all,
        vytde_all,
        confidences,
        vx_2dcas,
        vy_2dcas,
        cond_reprs,
    )


rand_coeffs = np.linspace(0, 1, num=10)

vx_all, vy_all, vxtde_all, vytde_all, confidences, vx_2dcas, vy_2dcas, cond_reprs = (
    parallel_loop_mp(rand_coeffs, N=N, processes=4)
)

data_file = "decoherence_data.npz"
np.savez(
    data_file,
    rand_coeffs=rand_coeffs,
    vx_all=vx_all,
    vy_all=vy_all,
    vxtde_all=vxtde_all,
    vytde_all=vytde_all,
    confidences=confidences,
    vx_2dcas=vx_2dcas,
    vy_2dcas=vy_2dcas,
    cond_reprs=cond_reprs,
)
print(f"Data saved to {data_file}")
