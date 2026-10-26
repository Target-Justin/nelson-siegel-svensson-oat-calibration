import numpy as np
import pandas as pd
import scipy.optimize as sp

from config import LAMBDA_MAX
from scipy.optimize import OptimizeResult
from scripts.common.nss_factors import build_factor_loadings_matrix
from scripts.calibration_methods.grid_search import grid_search
from scripts.calibration_methods.ols import ols, ols_weighted
from scripts.calibration_methods.ssr_scoring import build_ssr_score

def calculate_ssr(parameters : np.ndarray, zero_rate : pd.DataFrame, is_ns : bool, weights : list = None) -> float:
    """
    Objective function for joint optimization: SSR for a single vector
    holding both betas and lambda(s) (NS: [b1,b2,b3,lambda_1]; NSS:
    [b1,b2,b3,b4,lambda_1,lambda_2]), unlike ols which re-solves the betas
    internally.

    weights: optional weights array; SSR is weighted by it if provided,
    unweighted otherwise. Must match ZeroCouponRatePct order.

    Returns +inf outside the admissible region: lambda_1 <= 0, the last
    lambda above LAMBDA_MAX, or (NSS only) lambda_2 <= lambda_1 + 1.
    """

    zero_rate_matrix=zero_rate[["ZeroCouponRatePct"]].to_numpy()
    time_list=list(zero_rate["TimeToMaturity"])

    if is_ns:
        if 0< parameters[3] <= LAMBDA_MAX:
            factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, parameters[3], None)
            zero_rate_estimator_matrix=factor_loadings_matrix@parameters[:3]
        else:
            return np.inf

    else:
        if (parameters[4] + 1 < parameters[5]) and (parameters[4] > 0) and (parameters[5] <= LAMBDA_MAX):
            factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, parameters[4], parameters[5])
            zero_rate_estimator_matrix=factor_loadings_matrix@parameters[:4]
        else:
            return np.inf
    
    residual_matrix=zero_rate_matrix.flatten()-zero_rate_estimator_matrix

    if weights is not None:
        ssr=float(np.sum(weights * residual_matrix**2))
    else:
        ssr=float(np.sum(residual_matrix**2))

    return ssr

def sp_calibration(zero_rate : pd.DataFrame, weights : np.ndarray, is_ns : bool, is_weighted : bool, threshold : float, lambda_grids : tuple[np.ndarray, np.ndarray]) -> OptimizeResult:
    """
    Joint optimization equivalent of calibrate_nss: minimizes calculate_ssr
    over betas and lambda(s) at once via scipy's Nelder-Mead, instead of
    the two-stage lambda-search-then-OLS approach. Weighted by weights when
    is_weighted is True, consistently with calibrate_nss.

    The starting point reuses calibrate_nss's own building blocks:
    grid_search on lambda_grids for the lambdas (via the ssr_score built by
    build_ssr_score, so the search already reflects is_weighted), then ols
    or ols_weighted for the corresponding betas.

    Returns the scipy OptimizeResult, with x = [betas..., lambda(s)] (see
    unpack_scipy_result) and fun = SSR.
    Raises ValueError if no grid candidate satisfies the threshold.
    """

    ssr_score, _=build_ssr_score(zero_rate, weights, is_ns, is_weighted)

    ssr_weights=weights if is_weighted else None
    
    first_estimation_of_lambdas=grid_search(zero_rate, is_ns, ssr_score, threshold, lambda_grids[0], lambda_grids[1])
    if is_weighted:
        beta=ols_weighted(zero_rate, weights, is_ns, True, first_estimation_of_lambdas[0], first_estimation_of_lambdas[1])[0]
    else:
        beta=ols(zero_rate, is_ns, True, first_estimation_of_lambdas[0], first_estimation_of_lambdas[1])[0]

    if is_ns:
        return sp.minimize(calculate_ssr, np.concatenate((beta.flatten(), [first_estimation_of_lambdas[0]])), args=(zero_rate, is_ns, ssr_weights), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxfev": 10000})

    else:
        return sp.minimize(calculate_ssr, np.concatenate((beta.flatten(), first_estimation_of_lambdas)), args=(zero_rate, is_ns, ssr_weights), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxfev": 10000})