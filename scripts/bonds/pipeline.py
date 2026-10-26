import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, NSS_CONFIGS, THRESHOLDS, LAMBDA_GRIDS
from scripts.bonds.bond_metrics.yield_to_maturity import generate_yield_to_maturity
from scripts.bonds.bond_metrics.duration import generate_duration
from scripts.common.nss_pipeline import run_nss

def run_bonds() -> pd.DataFrame:
    """
    Run the four NS/NSS calibrations on the OAT zero-coupon curve.

    Reads the outputs of the bootstrap project in data/raw/bonds/, computes
    the yields to maturity and Macaulay durations (saved in
    data/processed/bonds/), weights each bond by 1/duration and calls run_nss
    for each configuration of NSS_CONFIGS (results in results/bonds/).

    Returns the concatenated parameters of the four configurations.
    Raises ValueError if a duration is missing or non-positive.
    """

    # --- Data loading ---
    bond=pd.read_csv(RAW_DIR["OAT"] /"dataset.csv")
    cashflow=pd.read_csv(RAW_DIR["OAT"] / "cashflow.csv")
    dirty_price=pd.read_csv(RAW_DIR["OAT"] / "dirty_price.csv")
    zero_rate=pd.read_csv(RAW_DIR["OAT"] / "zero_rate.csv")

    bond["Maturity"]=pd.to_datetime(bond["Maturity"])
    bond["EvaluationDate"]=pd.to_datetime(bond["EvaluationDate"])
    cashflow["Date"]=pd.to_datetime(cashflow["Date"])
    dirty_price["Maturity"]=pd.to_datetime(dirty_price["Maturity"])
    zero_rate["Maturity"]=pd.to_datetime(zero_rate["Maturity"])

    bond=bond.drop(columns=["Coupon","CleanPrice","IssueDate","AccrualStartDate"]).merge(
        dirty_price[["Bond","DirtyPrice"]], on="Bond", how="left")

    # --- Bond metrics ---
    yield_to_maturity=generate_yield_to_maturity(bond, cashflow)
    yield_to_maturity.to_csv(PROCESSED_DIR["OAT"] / "yield_to_maturity.csv",index=False)

    bond=bond.merge(yield_to_maturity[["Bond", "YieldToMaturity"]], on="Bond", how="left")

    duration=generate_duration(bond, cashflow)
    duration.to_csv(PROCESSED_DIR["OAT"] / "duration.csv", index=False)

    zero_rate=zero_rate.merge(duration[["Bond", "Duration"]], on="Bond", how="left", validate="one_to_one")

    if zero_rate["Duration"].isna().any() or (zero_rate["Duration"] <= 0).any():
        raise ValueError("Missing or non-positive duration: cannot compute 1/duration weights.")

    weights=1.0 / zero_rate["Duration"].to_numpy()

    parameters_list = []

    for is_ns, is_weighted in NSS_CONFIGS:
        parameters = run_nss(zero_rate, weights, is_ns, is_weighted, THRESHOLDS["OAT"], LAMBDA_GRIDS["OAT"], RESULTS_DIR["OAT"], "oat")
        parameters_list.append(parameters)

    parameters_df = pd.concat(parameters_list, ignore_index=True)
    return parameters_df