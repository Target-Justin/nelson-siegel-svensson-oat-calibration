import numpy as np
from typing import Callable

def adaptive_step_search(ssr_score : Callable[[float, float], float], lambda_1 : float, step : float, max_iteration : int) -> tuple[float, None]:
    """
    1-D local refinement of lambda_1 (NS only) around a starting point.

    Each iteration tries a move of size step in both directions, evaluated
    by ssr_score; a move to lambda_1 <= 0 is treated as infeasible (+inf),
    since ssr_score is not penalized in the NS case. Takes the improving
    move, or halves step if neither improves. Second return value is
    always None, for signature symmetry with nelder_mead.
    """
    
    ssr=ssr_score(lambda_1, None)
    for _ in range(max_iteration):
        ssr_minus=ssr_score(lambda_1-step, None) if lambda_1 - step > 0 else np.inf
        ssr_plus=ssr_score(lambda_1+step, None)
        if ssr_minus < ssr:
            lambda_1-=step
            ssr=ssr_minus
        elif ssr_plus < ssr:
            lambda_1+=step
            ssr=ssr_plus
        else:
            step=step/2
    return lambda_1, None