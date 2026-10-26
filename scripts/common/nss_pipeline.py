import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.calibration_methods.calibrate import calibrate_nss
from scripts.calibration_methods.scipy_calibration.scipy_calibrate import sp_calibration
from scripts.common.nss_factors import build_factor_loadings_matrix, unpack_scipy_result
from scripts.common.plot_zero_rates_curve import plot_zero_rates_curve

def run_nss(zero_rate: pd.DataFrame, weights: np.ndarray, is_ns: bool, is_weighted: bool, threshold: float, lambda_grids: tuple[np.ndarray, np.ndarray], output_dir: Path, name: str) -> pd.DataFrame:
    """
    Calibrate one NS/NSS configuration with both methods (two-step and
    scipy), export the results and return the parameters.

    zero_rate: one curve, with the columns Maturity, TimeToMaturity (years)
    and ZeroCouponRatePct (%), sorted by maturity.
    weights: 1D array aligned with the rows of zero_rate, only used when
    is_weighted is True.
    threshold, lambda_grids: grid search settings (see grid_search).
    output_dir, name: the files {name}_{model}_{weighting}_parameters.csv,
    _comparison.csv (fitted rates of both methods and their difference in bp)
    and .png are written in output_dir.

    Returns a 2-row DataFrame (TwoStep, Scipy) with the columns Name, Model,
    Weighting, Method, Beta1-Beta4, Lambda1, Lambda2 and SSR.
    """
    
    # --- Export ---
    model_name="ns" if is_ns else "nss"
    weighting_name="weighted" if is_weighted else "unweighted"
    file_name=f"{name}_{model_name}_{weighting_name}"

    output_dir.mkdir(parents=True, exist_ok=True)

    nss=calibrate_nss(zero_rate, weights, is_ns, is_weighted, threshold, lambda_grids)
    beta_matrix, lambda_1, lambda_2, zero_rate_estimator_matrix, ssr = nss

    scipy_result=sp_calibration(zero_rate, weights, is_ns, is_weighted, threshold, lambda_grids)
    ssr_scipy=float(scipy_result.fun)

    # Parameters: one row per method (two-step vs scipy), same columns for both
    beta_flat=beta_matrix.flatten()
    scipy_beta, scipy_lambda_1, scipy_lambda_2=unpack_scipy_result(scipy_result, is_ns)

    two_step_row={"Method": "TwoStep", "Beta1": beta_flat[0], "Beta2": beta_flat[1],
                    "Beta3": beta_flat[2], "Beta4": beta_flat[3] if not is_ns else None, 
                    "Lambda1": lambda_1, "Lambda2": lambda_2, "SSR": ssr}

    scipy_row={"Method": "Scipy", "Beta1": scipy_beta[0], "Beta2": scipy_beta[1],
                "Beta3": scipy_beta[2], "Beta4": scipy_beta[3] if not is_ns else None,
                "Lambda1": scipy_lambda_1, "Lambda2": scipy_lambda_2, "SSR": ssr_scipy}

    parameters = pd.DataFrame([two_step_row, scipy_row])
    parameters.insert(0, "Name", name)
    parameters.insert(1, "Model", model_name)
    parameters.insert(2, "Weighting", weighting_name)
    parameters.to_csv(output_dir / f"{file_name}_parameters.csv", index=False)

    # Comparison at the observed maturities
    time_list=list(zero_rate["TimeToMaturity"])
    scipy_factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, scipy_lambda_1, scipy_lambda_2)
    scipy_fitted_rates=(scipy_factor_loadings_matrix@scipy_beta).flatten()

    two_step_fitted_rates=zero_rate_estimator_matrix.flatten()
    observed_rates=zero_rate["ZeroCouponRatePct"].to_numpy()

    diff_bp=(scipy_fitted_rates-two_step_fitted_rates)*100

    comparison = pd.DataFrame({"Maturity": zero_rate["Maturity"], "TimeToMaturity": zero_rate["TimeToMaturity"],
                            "ObservedRatePct": observed_rates, "TwoStepFittedRatePct": two_step_fitted_rates,
                            "ScipyFittedRatePct": scipy_fitted_rates,"DiffBp": diff_bp})
    comparison.to_csv(output_dir / f"{file_name}_comparison.csv", index=False)

    # --- Console summary ---
    ssr_two_step = float(ssr)
    relative_ssr_diff = abs(ssr_scipy - ssr_two_step) / ssr_two_step
    abs_diff_bp = np.abs(diff_bp)

    print(f"\n=== {file_name.upper()}: two-step vs scipy ===")
    print(f"SSR two-step          : {ssr_two_step:.6f}")
    print(f"SSR scipy             : {ssr_scipy:.6f}")
    print(f"Relative SSR diff     : {relative_ssr_diff:.4%}")
    print(f"Max |fitted diff|     : {abs_diff_bp.max():.4f} bp")
    print(f"Mean |fitted diff|    : {abs_diff_bp.mean():.4f} bp")

    # --- Plot: fitted curve against the bootstrapped points ---
    fig=plot_zero_rates_curve(zero_rate, is_ns, is_weighted, lambda_1, lambda_2, beta_matrix, scipy_result)
    fig.savefig(output_dir / f"{file_name}.png", dpi=300, bbox_inches="tight")

    plt.close(fig)

    return parameters