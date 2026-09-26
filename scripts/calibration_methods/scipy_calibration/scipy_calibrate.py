import numpy as np
import pandas as pd
import scipy.optimize as sp
from scipy.optimize import OptimizeResult
from scripts.nss_factors import build_factor_loadings_matrix
from scripts.calibration_methods.grid_search import grid_search
from scripts.calibration_methods.ols import ols, ols_weighted
from scripts.calibration_methods.ssr_scoring import build_ssr_score

def calculate_ssr(parameters : np.ndarray, zero_rate : pd.DataFrame, is_ns : bool, weights : list = None) -> float:
    """
    Objective function for joint optimization: SSR for a single vector
    holding both betas and lambda(s) (NS: [b1,b2,b3,lambda_1]; NSS:
    [b1,b2,b3,b4,lambda_1,lambda_2]), unlike ols which re-solves the betas
    internally.

    weights: optional 1/duration array; SSR is weighted by it if provided,
    unweighted otherwise. Must match ZeroCouponRatePct order.

    Returns +inf if any lambda is non-positive, or (NSS only) if
    lambda_2 <= lambda_1 + 1.
    """

    zero_rate_matrix=zero_rate[["ZeroCouponRatePct"]].to_numpy()
    time_list=list(zero_rate["TimeToMaturity"])

    if is_ns:
        if parameters[3] > 0:
            factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, parameters[3], None)
            zero_rate_estimator_matrix=factor_loadings_matrix@parameters[:3]
        else:
            return np.inf

    else:
        if (parameters[4] + 1 < parameters[5]) and (parameters[4] > 0) and (parameters[5] > 0):
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

def sp_calibration(zero_rate : pd.DataFrame, duration : pd.DataFrame, is_ns : bool, is_weighted : bool, threshold : float) -> OptimizeResult:
    """
    Joint optimization equivalent of calibrate_nss: minimizes calculate_ssr
    over betas and lambda(s) at once via Nelder-Mead, instead of the
    two-stage lambda-search-then-OLS approach. Weighted by 1/duration when
    is_weighted is True, consistently with calculate_ssr and calibrate_nss.

    Starting point reuses calibrate_nss's own building blocks: grid_search
    for the lambdas (via the ssr_score built by build_ssr_score, so the
    search already reflects is_weighted), then ols or ols_weighted for the
    corresponding betas.

    Raises ValueError if any duration is zero (weighted case only).
    """

    ssr_score, _=build_ssr_score(zero_rate, duration, is_ns, is_weighted)

    weights=None

    if is_weighted:
            duration_array = duration["Duration"].to_numpy()
            if np.any(duration_array == 0):
                raise ValueError("Duration cannot be equal to zero.")
            weights = 1.0 / duration_array
    
    first_estimation_of_lambdas=grid_search(zero_rate, is_ns, ssr_score, threshold)
    if is_weighted:
        beta=ols_weighted(zero_rate, duration, is_ns, True, first_estimation_of_lambdas[0], first_estimation_of_lambdas[1])[0]
    else:
        beta=ols(zero_rate, is_ns, True, first_estimation_of_lambdas[0], first_estimation_of_lambdas[1])[0]

    if is_ns:
        return sp.minimize(calculate_ssr, np.concatenate((beta.flatten(), [first_estimation_of_lambdas[0]])), args=(zero_rate, is_ns, weights), method="Nelder-Mead")

    else:
        return sp.minimize(calculate_ssr, np.concatenate((beta.flatten(), first_estimation_of_lambdas)), args=(zero_rate, is_ns, weights), method="Nelder-Mead")