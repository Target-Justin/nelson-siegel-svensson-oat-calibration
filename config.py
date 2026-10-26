import numpy as np
import QuantLib as ql

from pathlib import Path

RAW_DIR = {"OAT" : Path("data/raw/bonds"), "OIS" : Path("data/raw/ois")}
PROCESSED_DIR = {"OAT" : Path("data/processed/bonds"), "OIS" : Path("data/processed/ois")}
RESULTS_DIR = {"OAT" : Path("results/bonds"), "OIS" : Path("results/ois")}

NSS_CONFIGS = [(True, False), (True, True), (False, False), (False, True)]

CALENDAR_SCHEDULE_CONVENTIONS = {"OAT" : ql.NullCalendar(), "ESTR" : ql.TARGET(), "SOFR" : ql.UnitedStates(ql.UnitedStates.SOFR)}
SETTLEMENT_CALENDAR_CONVENTIONS = {"OAT" : ql.TARGET()}
SETTLEMENT_DAYS_CONVENTIONS = {"OAT" : 2, "OIS" : 2}
OVERNIGHT_INDEX_CONVENTIONS = {"ESTR" : ql.Estr(), "SOFR" : ql.Sofr()}
BUSINESS_DAY_CONVENTIONS = {"OAT" : ql.Unadjusted, "OIS" : ql.ModifiedFollowing}
DATE_GENERATION_CONVENTIONS = {"OAT" : ql.DateGeneration.Backward, "OIS" : ql.DateGeneration.Forward}
PAYMENT_PERIOD_CONVENTIONS = {"OIS" : ql.Period(ql.Annual)}
END_OF_MONTH_CONVENTIONS = {"OAT" : False, "OIS" : False}

LAMBDA_GRIDS = {"OAT" : (np.linspace(0.10, 2.40, 47), np.linspace(2.50, 5.45, 60)),
                "OIS" : (np.geomspace(0.05, 5.0, 60), np.geomspace(2.0, 25.0, 60))}
LAMBDA_MAX = 30.0
THRESHOLDS = {"OAT" : 0.9, "OIS" : 0.99}
MIN_BID_ASK_SPREAD_BP = 0.5