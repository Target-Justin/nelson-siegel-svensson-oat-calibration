import numpy as np
import pandas as pd
from scripts.nss_factors import build_factor_loadings_matrix
from typing import Optional

def ols(zero_rate : pd.DataFrame, is_ns : bool, return_parameters : bool, lambda_1 : float, lambda_2 : Optional[float]) -> float | tuple[np.ndarray, float, float | None, np.ndarray]:
    """
    Unweighted OLS fit of the NS/NSS betas for fixed lambda(s).

    Used both as an objective (return_parameters=False -> SSR only, for
    the lambda search) and as the final estimation step
    (return_parameters=True -> beta_matrix, lambdas, fitted rates).
    """

    zero_rate_matrix=zero_rate[["ZeroCouponRatePct"]].to_numpy()
    time_list=list(zero_rate["TimeToMaturity"])

    factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, lambda_1, lambda_2)

    beta_matrix,_,_,_=np.linalg.lstsq(factor_loadings_matrix, zero_rate_matrix, rcond=None)
    
    zero_rate_estimator_matrix=factor_loadings_matrix@beta_matrix
    residual_matrix=zero_rate_matrix-zero_rate_estimator_matrix

    ssr=float(np.sum(residual_matrix**2))

    if return_parameters:
        return beta_matrix, lambda_1, lambda_2, zero_rate_estimator_matrix, ssr
    return ssr

def ols_penalized(zero_rate : pd.DataFrame, is_ns : bool, lambda_1 : float, lambda_2 : float) -> float | tuple[np.ndarray, float, float, np.ndarray]:
    """
    Wraps ols with two feasibility guards, returning +inf when violated so
    nelder_mead avoids that region:
    - lambda_1 <= 0 (lambda must be strictly positive)
    - lambda_2 <= lambda_1 + 1 (NSS identifiability constraint between the
      two curvature factors)
    """

    if lambda_2 <= lambda_1 + 1 or lambda_1 <= 0:
        return np.inf
    return ols(zero_rate, is_ns, False, lambda_1, lambda_2)

def ols_weighted(zero_rate : pd.DataFrame, duration : pd.DataFrame, is_ns : bool, return_parameters : bool, lambda_1 : float, lambda_2 : float) -> float | tuple[np.ndarray, float, float, np.ndarray]:
    """
    Weighted counterpart of ols_penalized: wraps ols_weighted with the same
    two feasibility guards (lambda_1 <= 0, and lambda_2 <= lambda_1 + 1 for
    NSS identifiability), returning +inf when either is violated so
    nelder_mead avoids that region.
    """
    
    zero_rate_matrix=zero_rate[["ZeroCouponRatePct"]].to_numpy()
    time_list=list(zero_rate["TimeToMaturity"])
    duration_array = duration["Duration"].to_numpy()

    if np.any(duration_array == 0):
        raise ValueError("Duration cannot be equal to zero.")

    weights = 1.0 / duration_array

    factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, lambda_1, lambda_2)
   
    sqrt_w=np.sqrt(weights).reshape(-1, 1)
    weighted_factor_loadings_matrix=factor_loadings_matrix*sqrt_w
    zero_rate_matrix_w=zero_rate_matrix*sqrt_w

    beta_matrix,_,_,_=np.linalg.lstsq(weighted_factor_loadings_matrix, zero_rate_matrix_w, rcond=None)

    zero_rate_estimator_matrix=factor_loadings_matrix@beta_matrix
    residual_matrix=zero_rate_matrix-zero_rate_estimator_matrix

    ssr = float(np.sum(weights * residual_matrix[:, 0]**2))

    if return_parameters:
        return beta_matrix, lambda_1, lambda_2, zero_rate_estimator_matrix, ssr
    return ssr

def ols_weighted_penalized(zero_rate : pd.DataFrame, duration : pd.DataFrame, is_ns : bool, lambda_1 : float, lambda_2 : float) -> float | tuple[np.ndarray, float, float, np.ndarray]:
    """
    Weighted counterpart of ols_penalized: wraps ols_weighted with the NSS
    identifiability constraint lambda_2 > lambda_1 + 1, returning +inf when
    violated so nelder_mead avoids that region.
    """

    if lambda_2 <= lambda_1 + 1 or lambda_1 <= 0:
        return np.inf
    return ols_weighted(zero_rate, duration, is_ns, False, lambda_1, lambda_2)