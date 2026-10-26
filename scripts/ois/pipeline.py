import pandas as pd

from config import RAW_DIR, RESULTS_DIR, NSS_CONFIGS, THRESHOLDS, LAMBDA_GRIDS
from scripts.common.nss_pipeline import run_nss
from scripts.ois.loader import load_ois_curve

def run_ois() -> pd.DataFrame:
    """
    Run the four NS/NSS calibrations on every OIS curve of
    data/raw/ois/zero_rate.csv (long format produced by the bootstrap
    project, one curve per Curve and EvaluationDate).

    Each curve is weighted by 1/max(bid-ask spread, MIN_BID_ASK_SPREAD_BP)²
    (see load_ois_curve). Results go to results/ois/<curve>/ and all the
    parameters are gathered in results/ois/ois_parameters_all.csv. Failed
    calibrations (ValueError, RuntimeError) are printed at the end instead
    of stopping the loop.

    Returns the gathered parameters, with the extra columns Curve and Date.
    Raises FileNotFoundError if the input file is missing, and ValueError if
    every calibration fails.
    """
        
    # --- Data loading ---
    zero_rate_all=pd.read_csv(RAW_DIR["OIS"] / "zero_rate.csv")
    zero_rate_all["Maturity"]=pd.to_datetime(zero_rate_all["Maturity"])

    parameters_list = []
    errors_list = []   

    for (curve, evaluation_date), curve_df in zero_rate_all.groupby(["Curve", "EvaluationDate"]):

        zero_rate, weights = load_ois_curve(curve_df)

        name = f"{curve.lower()}_{evaluation_date}"
        output_dir = RESULTS_DIR["OIS"] / curve.lower()

        for is_ns, is_weighted in NSS_CONFIGS:

            try:

                parameters = run_nss(zero_rate, weights, is_ns, is_weighted, THRESHOLDS["OIS"], LAMBDA_GRIDS["OIS"], output_dir, name)
                parameters.insert(0, "Curve", curve)
                parameters.insert(1, "Date", pd.to_datetime(evaluation_date))
                parameters_list.append(parameters)

            except (ValueError, RuntimeError) as error:

                errors_list.append((name, is_ns, is_weighted, str(error)))

    if errors_list:
        for name, is_ns, is_weighted, message in errors_list:
            print(f"FAILED {name} (ns={is_ns}, weighted={is_weighted}): {message}")

    if not parameters_list:
        raise ValueError("All OIS calibrations failed, see the errors above")

    parameters = pd.concat(parameters_list, ignore_index=True)
    parameters.to_csv(RESULTS_DIR["OIS"] / "ois_parameters_all.csv", index=False)
    return parameters