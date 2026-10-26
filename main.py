import argparse as ap
import pandas as pd

from config import RAW_DIR
from scripts.bonds.pipeline import run_bonds
from scripts.ois.pipeline import run_ois

def main():
    """
    Run the NS/NSS calibration pipelines selected on the command line.

    Usage: python main.py [bonds|ois|all]. Without argument, all the
    pipelines are run. The OIS step is skipped if data/raw/ois/zero_rate.csv
    is missing (the OIS data are not published).
    """

    file_path=RAW_DIR["OIS"] / "zero_rate.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No OIS input found at {file_path}. The Bloomberg data are not published: see the Data Availability section of the README.")
    zero_rate_all=pd.read_csv(file_path)

    parser = ap.ArgumentParser(description = "Run a bootstrap pipeline")
    parser.add_argument("target", choices = ["bonds","ois","all"], nargs = "?", default = "all")
    args = parser.parse_args()

    if args.target == "bonds":
        run_bonds()
    elif args.target == "ois":
        run_ois()
    elif args.target == "all":
        run_bonds()
        run_ois()

if __name__ == "__main__":
    main()