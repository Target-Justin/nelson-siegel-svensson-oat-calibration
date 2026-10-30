import argparse as ap

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

    parser = ap.ArgumentParser(description = "Run a bootstrap pipeline")
    parser.add_argument("target", choices = ["bonds","ois","all"], nargs = "?", default = "all")
    args = parser.parse_args()

    if args.target == "bonds":
        run_bonds()
    elif args.target == "ois":
        run_ois()
    elif args.target == "all":
        run_bonds()
        ois_input = RAW_DIR["OIS"] / "zero_rate.csv"
        if ois_input.exists():
            run_ois()
        else:
            print(f"OIS step skipped: {ois_input} not found (the Bloomberg data are not published, see the Data Availability section of the README).")

if __name__ == "__main__":
    main()