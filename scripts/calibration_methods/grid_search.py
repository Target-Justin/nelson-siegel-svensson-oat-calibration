import pandas as pd
import numpy as np
from scripts.nss_factors import x_1, x_2, x_3
from typing import Callable

def grid_search(zero_rate : pd.DataFrame, is_ns : bool, ssr_score : Callable[[float, float], float], threshold : float) -> tuple[float, float | None]:
    """
    Coarse grid search for a starting lambda (NS) or (lambda_1, lambda_2)
    pair (NSS), to initialize the local refinement in calibrate_nss.

    Candidates are drawn from a fixed grid and evaluated by ssr_score; a
    candidate is skipped if any pair of its factor loadings has a
    correlation exceeding threshold (avoids an ill-conditioned factor
    loadings matrix). lambda_2 is None when is_ns is True.

    Raises ValueError if no candidate satisfies the threshold.
    """

    time_list=list(zero_rate["TimeToMaturity"])

    ssr=np.inf

    lambda_1,lambda_2=None,None
    lambda_1_grid = np.linspace(0.10, 2.40, 47)
    lambda_2_grid = np.linspace(2.50, 5.45, 60)

    for lambda_1_candidate in lambda_1_grid:

        x_1_values=np.array([x_1(t, lambda_1_candidate) for t in time_list])
        x_2_values=np.array([x_2(t, lambda_1_candidate) for t in time_list])

        cor_12=np.corrcoef(x_1_values, x_2_values)[0, 1]

        if abs(cor_12)>=threshold:
            continue

        if is_ns:
            new_ssr=ssr_score(lambda_1_candidate, None)

            if new_ssr < ssr:
                lambda_1, lambda_2, ssr= lambda_1_candidate, None, new_ssr

        else:

            for lambda_2_candidate in lambda_2_grid:

                if lambda_2_candidate<=lambda_1_candidate+1:
                    continue

                x_3_values=np.array([x_3(t, lambda_2_candidate) for t in time_list])

                cor_13=np.corrcoef(x_1_values, x_3_values)[0, 1]
                cor_23=np.corrcoef(x_2_values, x_3_values)[0, 1]

                if (abs(cor_13)>=threshold) or (abs(cor_23)>=threshold):
                    continue

                new_ssr=ssr_score(lambda_1_candidate, lambda_2_candidate)

                if new_ssr < ssr:
                    lambda_1, lambda_2, ssr= lambda_1_candidate, lambda_2_candidate, new_ssr

    if lambda_1 is None:
        raise ValueError(f"No (lambda_1, lambda_2) pair satisfies the correlation "
                        f"threshold={threshold}. Try increasing the threshold.")

    return lambda_1,lambda_2