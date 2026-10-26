import pandas as pd
import numpy as np
import QuantLib as ql

from config import SETTLEMENT_CALENDAR_CONVENTIONS, SETTLEMENT_DAYS_CONVENTIONS
from scripts.common.utils import to_ql_date, get_day_counter
from scripts.bonds.bond_metrics.numerical_methods.newton_raphson import newton_raphson

def generate_yield_to_maturity(bond : pd.DataFrame, cashflow : pd.DataFrame) -> pd.DataFrame:
    """
    Solve each bond's YTM by Newton-Raphson: the continuously-compounded
    rate that discounts its cashflows (from settlement = evaluation date + 2
    calendar days) to the observed dirty price. Initial guess: 3%.

    Returns a DataFrame with the columns Bond, Maturity and YieldToMaturity
    (decimal).
    """
    
    yield_to_maturity=[]

    for bond_data in bond.itertuples():

        bd=bond_data.Bond
        evaluation_date=to_ql_date(bond_data.EvaluationDate)
        settlement_date=SETTLEMENT_CALENDAR_CONVENTIONS["OAT"].advance(evaluation_date, SETTLEMENT_DAYS_CONVENTIONS["OAT"])
        maturity=bond_data.Maturity
        dp=bond_data.DirtyPrice
        day_counter=get_day_counter(bond_data.DayCount)

        bond_cashflow=cashflow[cashflow.Bond == bond_data.Bond]

        cf_amounts = bond_cashflow["Cashflow"].to_numpy()

        dates = bond_cashflow["Date"].map(to_ql_date)
        times_to_cashflow = np.array([day_counter.yearFraction(settlement_date, date) for date in dates])

        def eq(y):
            return -dp + np.sum(cf_amounts * np.exp(-y * times_to_cashflow))

        def deq(y):
            return -np.sum(times_to_cashflow*cf_amounts*np.exp(-y*times_to_cashflow))

        yield_to_maturity.append({"Bond" : bd, "Maturity" : maturity, "YieldToMaturity" : newton_raphson(eq, deq, 0.03)})

    return pd.DataFrame(yield_to_maturity)