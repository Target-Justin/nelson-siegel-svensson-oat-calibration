import numpy as np

from config import LAMBDA_MAX
from typing import Callable

def adaptive_step_search(ssr_score : Callable[[float, float], float], lambda_1 : float, step : float, max_iteration : int, x_rtol : float) -> tuple[float, None]:
    """
    1-D local refinement of lambda_1 (NS only) around a starting point.

    Each iteration tries a move of size step in both directions, evaluated
    by ssr_score. A move to lambda_1 <= 0 or above LAMBDA_MAX is treated as
    infeasible (+inf), since ssr_score is not penalized in the NS case.
    Takes the improving move, or halves step if neither improves.

    Stops when step <= x_rtol * lambda_1. The second return value is always
    None, for signature symmetry with nelder_mead.
    Raises RuntimeError if not converged after max_iteration iterations.
    """
    
    ssr=ssr_score(lambda_1, None)
    for _ in range(max_iteration):
        if step <= x_rtol * lambda_1:
            return lambda_1, None
        ssr_minus=ssr_score(lambda_1-step, None) if lambda_1 - step > 0 else np.inf
        ssr_plus=ssr_score(lambda_1+step, None) if lambda_1 + step <= LAMBDA_MAX else np.inf
        if ssr_minus < ssr:
            lambda_1-=step
            ssr=ssr_minus
        elif ssr_plus < ssr:
            lambda_1+=step
            ssr=ssr_plus
        else:
            step=step/2
    raise RuntimeError("The adaptive step search does not converge.")