import numpy as np
import pandas as pd

from config import MIN_BID_ASK_SPREAD_BP
from pathlib import Path

def load_ois_curve(curve_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare one OIS curve for the calibration.

    Sorts the curve by TimeToMaturity (ACT/360 years from settlement) and
    computes the bid-ask weights 1 / max(BidAskSpreadBp, MIN_BID_ASK_SPREAD_BP)²,
    aligned with the sorted rows. The floor avoids infinite weights on
    zero-spread quotes.

    Returns (sorted curve, weights).
    Raises ValueError if a TimeToMaturity is not strictly positive.
    """

    curve_df = curve_df.sort_values("TimeToMaturity").reset_index(drop = True)

    if (curve_df["TimeToMaturity"] <= 0).any():
        raise ValueError("Non-positive TimeToMaturity in OIS curve: the NS/NSS factor loadings are not defined")

    spread=curve_df["BidAskSpreadBp"].clip(lower=MIN_BID_ASK_SPREAD_BP).to_numpy()
    weights=1.0/spread**2

    return curve_df, weights