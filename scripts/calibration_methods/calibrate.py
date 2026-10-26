import numpy as np
import pandas as pd

from scripts.calibration_methods.ssr_scoring import build_ssr_score
from scripts.calibration_methods.grid_search import grid_search
from scripts.calibration_methods.nelder_mead import nelder_mead
from scripts.calibration_methods.local_search import adaptive_step_search
from scripts.calibration_methods.ols import ols, ols_weighted


def calibrate_nss(zero_rate : pd.DataFrame, weights : np.ndarray, is_ns : bool, is_weighted : bool , threshold : float, lambda_grids : tuple[np.ndarray, np.ndarray]) -> float | tuple[np.ndarray, float, float | None, np.ndarray]:
    """
    Two-stage calibration of a Nelson-Siegel (or Svensson) curve.

    Stage 1 (lambda selection): build_ssr_score fixes the OLS objective
    (weighted by weights or not, depending on is_weighted) and its
    penalized variant (+inf outside the admissible region). A coarse grid
    search on lambda_grids followed by a local refinement
    (adaptive_step_search for NS, nelder_mead for NSS) then finds the
    lambda(s) minimizing that objective, with lambda_2 <= LAMBDA_MAX.

    Stage 2 (beta estimation): with lambdas fixed, betas are recovered in
    closed form by (weighted, if requested) OLS, consistently with the
    objective used in stage 1.

    threshold: correlation ceiling used in the grid search to discard
    lambda candidates whose factor loadings are too collinear.

    Returns (beta_matrix, lambda_1, lambda_2, fitted_rates, ssr), with
    lambda_2 = None for NS.
    Raises ValueError if no grid candidate satisfies the threshold, and
    RuntimeError if the local refinement does not converge.
    """

    ssr_score, penalized_ssr_score= build_ssr_score(zero_rate, weights, is_ns, is_weighted)

    first_estimation_of_lambdas=grid_search(zero_rate, is_ns, ssr_score, threshold, lambda_grids[0], lambda_grids[1])
    if is_ns:
        estimation_of_lambdas=adaptive_step_search(ssr_score, first_estimation_of_lambdas[0], 0.025, 10000, 1e-6)
    else:
        estimation_of_lambdas=nelder_mead(penalized_ssr_score, first_estimation_of_lambdas[0], first_estimation_of_lambdas[1], 0.10, 1e-6, 1e-10, 10000)
    if is_weighted:
        return ols_weighted(zero_rate, weights, is_ns, True, estimation_of_lambdas[0], estimation_of_lambdas[1])
    else:
        return ols(zero_rate, is_ns, True, estimation_of_lambdas[0], estimation_of_lambdas[1])