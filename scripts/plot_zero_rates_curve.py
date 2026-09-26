import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
from scripts.nss_factors import build_factor_loadings_matrix, unpack_scipy_result

def plot_zero_rates_curve(zero_rate: pd.DataFrame, is_ns: bool, is_weighted : bool, lambda_1: float, lambda_2: float, beta_matrix: np.ndarray, scipy_result : OptimizeResult) -> plt.Figure:
    """
    Plot bootstrapped zero-coupon rates against the fitted NS/NSS curve.

    The curve is evaluated on a 300-point grid spanning the observed
    maturity range, using the calibrated lambda(s) and beta_matrix.
    """
    
    time_grid=np.linspace(zero_rate["TimeToMaturity"].min(), zero_rate["TimeToMaturity"].max(), 300)

    factor_loadings_matrix=build_factor_loadings_matrix(time_grid, is_ns, lambda_1, lambda_2)

    fitted_rates=(factor_loadings_matrix @ beta_matrix).flatten()

    scipy_beta, scipy_lambda_1, scipy_lambda_2 = unpack_scipy_result(scipy_result, is_ns)

    scipy_factor_loadings_matrix=build_factor_loadings_matrix(time_grid, is_ns, scipy_lambda_1, scipy_lambda_2)

    scipy_fitted_rates=(scipy_factor_loadings_matrix@scipy_beta).flatten()

    difference=scipy_fitted_rates-fitted_rates

    fig, (ax1, ax2)=plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    model_name="NS" if is_ns else "NSS"
    weighting_name="Weighted" if is_weighted else "Unweighted"

    ax1.scatter(zero_rate["TimeToMaturity"], zero_rate["ZeroCouponRatePct"],
               color="black", s=25, zorder=3, label="Observed zero coupon rates")
    ax1.plot(time_grid, fitted_rates, color="tab:red", linewidth=1.8,
            label=f"{weighting_name} {model_name} two step calibrated curve")
    ax1.plot(time_grid, scipy_fitted_rates, color="tab:blue", linewidth=1.8,
            label=f"{weighting_name} {model_name} scipy calibrated curve")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Zero coupon rates (%)")
    ax1.set_title("Comparison of zero coupon curve calibration")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.plot(time_grid, difference, color="tab:purple", linewidth=1.5)
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("SciPy - two step calibration\n(pp)")
    ax2.set_title("Difference between fitted curves")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    return fig