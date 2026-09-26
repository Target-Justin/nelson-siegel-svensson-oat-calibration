import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.bond_metrics.yield_to_maturity import generate_yield_to_maturity
from scripts.bond_metrics.duration import generate_duration
from scripts.calibration_methods.calibrate import calibrate_nss
from scripts.calibration_methods.scipy_calibration.scipy_calibrate import sp_calibration
from scripts.nss_factors import build_factor_loadings_matrix, unpack_scipy_result
from scripts.plot_zero_rates_curve import plot_zero_rates_curve

# --- Data loading ---
bond=pd.read_csv("data/raw/dataset.csv")
cashflow=pd.read_csv("data/raw/cashflow.csv")
dirty_price=pd.read_csv("data/raw/dirty_price.csv")
zero_rate=pd.read_csv("data/raw/zero_rate.csv")

bond["Maturity"]=pd.to_datetime(bond["Maturity"])
bond["EvaluationDate"]=pd.to_datetime(bond["EvaluationDate"])
cashflow["Date"]=pd.to_datetime(cashflow["Date"])
dirty_price["Maturity"]=pd.to_datetime(dirty_price["Maturity"])
zero_rate["Maturity"]=pd.to_datetime(zero_rate["Maturity"])

bond=bond.drop(columns=["Coupon","CleanPrice","IssueDate","AccrualStartDate"]).merge(dirty_price[["Bond","DirtyPrice"]], on="Bond", how="left")

# --- Bond metrics ---
yield_to_maturity=generate_yield_to_maturity(bond, cashflow)
yield_to_maturity.to_csv("data/processed/yield_to_maturity.csv",index=False)

bond=bond.merge(yield_to_maturity[["Bond", "YieldToMaturity"]], on="Bond", how="left")

duration=generate_duration(bond, cashflow)
duration.to_csv("data/processed/duration.csv", index=False)

# --- Calibration settings ---
is_ns=True          # True: Nelson-Siegel (3 betas, 1 lambda); False: Svensson (4 betas, 2 lambdas)
is_weighted=True    # weighted OLS: better fit on short/medium maturities, at the cost of the long end

# 0.5 = max correlation threshold between factors, used to discard lambda
# candidates that are too correlated during the grid search (same threshold reused by sp_calibration)
nss=calibrate_nss(zero_rate, duration, is_ns, is_weighted, 0.5)
beta_matrix, lambda_1, lambda_2, zero_rate_estimator_matrix, ssr = nss

scipy_result=sp_calibration(zero_rate, duration, is_ns, is_weighted, 0.5)
ssr_scipy=float(scipy_result.fun)

values=list(beta_matrix.flatten())+[lambda_1, lambda_2, ssr]

# --- Export ---
model_name="ns" if is_ns else "nss"
weighting_name="weighted" if is_weighted else "unweighted"

# Parameters: one row per method (two-step vs scipy), same columns for both
beta_flat=beta_matrix.flatten()
scipy_beta, scipy_lambda_1, scipy_lambda_2=unpack_scipy_result(scipy_result, is_ns)
scipy_beta_flat=scipy_beta.flatten() if hasattr(scipy_beta, "flatten") else scipy_beta

two_step_row={"Method": "TwoStep", "Beta1": beta_flat[0], "Beta2": beta_flat[1],
                "Beta3": beta_flat[2], "Beta4": beta_flat[3] if not is_ns else None, 
                "Lambda1": lambda_1, "Lambda2": lambda_2, "SSR": ssr}

scipy_row={"Method": "Scipy", "Beta1": scipy_beta_flat[0], "Beta2": scipy_beta_flat[1],
            "Beta3": scipy_beta_flat[2], "Beta4": scipy_beta_flat[3] if not is_ns else None,
            "Lambda1": scipy_lambda_1, "Lambda2": scipy_lambda_2, "SSR": ssr_scipy}

pd.DataFrame([two_step_row, scipy_row]).to_csv(f"results/{model_name}_{weighting_name}_parameters.csv", index=False)

# Comparison at the actual bond maturities
time_list=list(zero_rate["TimeToMaturity"])
scipy_factor_loadings_matrix=build_factor_loadings_matrix(time_list, is_ns, scipy_lambda_1, scipy_lambda_2)
scipy_fitted_rates=(scipy_factor_loadings_matrix@scipy_beta).flatten()

two_step_fitted_rates=zero_rate_estimator_matrix.flatten()
observed_rates=zero_rate["ZeroCouponRatePct"].to_numpy()

diff_bp=(scipy_fitted_rates-two_step_fitted_rates)*100

comparison = pd.DataFrame({"Maturity": zero_rate["Maturity"], "TimeToMaturity": zero_rate["TimeToMaturity"],
                           "ObservedRatePct": observed_rates, "TwoStepFittedRatePct": two_step_fitted_rates,
                           "ScipyFittedRatePct": scipy_fitted_rates,"DiffBp": diff_bp})
comparison.to_csv(f"results/{model_name}_{weighting_name}_comparison.csv", index=False)

# --- Plot: fitted curve against the bootstrapped points ---
fig=plot_zero_rates_curve(zero_rate, is_ns, is_weighted, lambda_1, lambda_2, beta_matrix, scipy_result)
fig.savefig(f"results/{model_name}_{weighting_name}.png", dpi=300, bbox_inches="tight")

plt.show()