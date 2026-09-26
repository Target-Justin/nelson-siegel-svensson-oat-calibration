import pandas as pd
from scripts.calibration_methods.ols import ols, ols_weighted, ols_penalized, ols_weighted_penalized
from typing import Callable

def build_ssr_score(zero_rate : pd.DataFrame, duration : pd.DataFrame, is_ns : bool, is_weighted : bool) -> tuple[Callable[[float, float], float], Callable[[float, float], float]]:
    """
    Builds the two OLS-SSR objective functions used by calibrate_nss to
    select lambda(s), with zero_rate, duration, is_ns and is_weighted fixed.

    Returns (ssr_score, penalized_ssr_score):
    - ssr_score(lambda_1, lambda_2): the raw SSR, weighted by 1/duration if
    is_weighted is True and unweighted otherwise. lambda_2 must be None
      when is_ns is True. Used by grid_search and adaptive_step_search.
    - penalized_ssr_score(lambda_1, lambda_2): same objective, returning
      +inf when the NSS identifiability constraint lambda_2 > lambda_1 + 1
      is violated. Used by nelder_mead (NSS only).
    """
    
    def ssr_score(lambda_1 : float, lambda_2 : float|None) -> float:
        if is_weighted:
            return ols_weighted(zero_rate, duration, is_ns, False, lambda_1, lambda_2)
        else:
            return ols(zero_rate, is_ns, False, lambda_1, lambda_2)

    def penalized_ssr_score(lambda_1 : float, lambda_2 : float) -> float:
        if is_weighted:
            return ols_weighted_penalized(zero_rate, duration, is_ns, lambda_1, lambda_2)
        else:
            return ols_penalized(zero_rate, is_ns, lambda_1, lambda_2)

    return ssr_score, penalized_ssr_score