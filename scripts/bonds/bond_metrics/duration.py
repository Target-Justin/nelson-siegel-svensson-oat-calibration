import numpy as np
import pandas as pd
import QuantLib as ql

from config import SETTLEMENT_CALENDAR_CONVENTIONS, SETTLEMENT_DAYS_CONVENTIONS
from scripts.common.utils import to_ql_date, get_day_counter

def generate_duration(bond : pd.DataFrame, cashflow : pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Macaulay duration of each bond, from settlement (evaluation
    date + 2 TARGET business days) to each cashflow date, discounting at the
    bond's own YTM under continuous compounding.

    Requires DirtyPrice and YieldToMaturity already merged into bond.
    Returns a DataFrame with the columns Bond, Maturity and Duration (years).
    Raises ValueError if any DirtyPrice is zero.
    """
    
    duration = []

    for bond_data in bond.itertuples():

        bd=bond_data.Bond
        maturity=bond_data.Maturity
        evaluation_date=to_ql_date(bond_data.EvaluationDate)
        settlement_date=SETTLEMENT_CALENDAR_CONVENTIONS["OAT"].advance(evaluation_date, SETTLEMENT_DAYS_CONVENTIONS["OAT"], ql.Days)
        dp=bond_data.DirtyPrice
        ytm=bond_data.YieldToMaturity
        day_counter=get_day_counter(bond_data.DayCount)

        duration_calculated=0

        if dp==0:

            raise ValueError("One of the dirty price is zero. Can't divide by zero.")

        bond_cashflow=cashflow[cashflow.Bond == bond_data.Bond]

        for cf in bond_cashflow.itertuples():

            date=to_ql_date(cf.Date)
            cf_amount=cf.Cashflow

            time_to_cashflow=day_counter.yearFraction(settlement_date,date)

            duration_calculated+=time_to_cashflow*cf_amount*np.exp(-time_to_cashflow*ytm)
            
        duration_calculated=duration_calculated/dp

        duration.append({"Bond" : bd, "Maturity" : maturity, "Duration" : duration_calculated})

    return pd.DataFrame(duration)