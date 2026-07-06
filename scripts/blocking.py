# blocking.py

import re
from typing import Optional
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import math

# Plateau picking quantities.
MINN = 8 # At later levels if we have too few blocks the data can become noist.
NEXT = 1 # Require the next N levels to be consistent within given error bars.
PLATEAUTOL = 0.25 # How much error can change between levels before it is not a plateau.

def parse() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type = Path,
        help = "Path to the noci-rs output file.",
    )
    parser.add_argument(
        "--start",
        type = int,
        default = None,
        help = (
            "First iteration included in the blocking analysis. "
            "If omitted, use the first iteration at which the shift is active."
        ),
    )
    return parser.parse_args()

def extract(path: Path) -> pd.DataFrame:
    """
    Extract the stochastic QMC table.
    """
    floatPattern = (
        r"[+-]?"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
    )
    
    pattern = re.compile(
        rf"^\s*"
        rf"(?P<iter>\d+)\s+"
        rf"(?P<eproj>{floatPattern})\s+"
        rf"(?P<ecorr>{floatPattern})\s+"
        rf"(?P<shift>{floatPattern})\s+"
        rf"(?P<nw>{floatPattern})\s+"
        rf"(?P<nref>{floatPattern})\s+"
        rf"(?P<nsampled>{floatPattern})\s+"
        rf"(?P<nsampledo>\d+)"
        rf"\s*$"
    )

    rows = []

    with open(path, "r") as output:
        for line in output:
            match = pattern.match(line)

            if match is None:
                continue

            rows.append(
                (
                    int(match.group("iter")),
                    float(match.group("eproj")),
                    float(match.group("ecorr")),
                    float(match.group("shift")),
                    float(match.group("nw")),
                    float(match.group("nref")),
                    float(match.group("nsampled")),
                    int(match.group("nsampledo")),
                )
            )

    return pd.DataFrame(
        rows,
        columns = [
            "iter",
            "eproj",
            "ecorr",
            "shift",
            "nw",
            "nref",
            "nsampled",
            "nsampledo",
        ],
    )

def prepareObservables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived observables used in blocking analysis.
    """
    df = df.copy()

    referenceEnergy = np.nanmedian(
        df["eproj"] - df["ecorr"]
    )

    shiftActive = ~np.isclose(
        df["shift"].to_numpy(dtype = float),
        0.0,
    )

    df["shiftcorr"] = np.where(
        shiftActive,
        df["shift"] - referenceEnergy,
        np.nan,
    )

    return df 

def blocking(xi) -> pd.DataFrame:
    levels = []

    level = 0
    while xi.size >= 2:
        n = xi.size
        xbar = xi.mean()

        # c_0 = 1 / n \sum_{k = 1}^n (x_k - \bar{x})^2. Equation 8 of Flyvbjerg-Petersen with t = 0.
        c0 = (1 / n) * ((xi - xbar) ** 2).sum()

        # Variance of the sample mean \sigma^2(\bar{x}). Equation 26 of Flyvbjerg-Petersen.
        # \sigma^2(m) = \langle c_0 / (n - 1) \rangle.
        sigma2 = c0 / (n - 1)
        sigma = math.sqrt(sigma2)

        # Error of the above estimator. Equation 28 of Flyvbjerg-Petersen.
        # \sigma^2(m) \approx (c'_0 / (n' - 1)) \pm \sqrt{(2 / (n' - 1))} (c'_0 / (n' - 1)).
        dsigma2 = math.sqrt(2.0 / (n - 1)) * sigma2
        dsigma = sigma / math.sqrt(2.0 * (n - 1))

        levels.append((level, n, xbar, c0, sigma2, sigma, dsigma2, dsigma))
        
        # If the data set is odd we must remove 1 element for this to work.
        if n % 2 == 1:
            xi = xi[:-1]
            n -= 1 
        # X'_i = (1 / 2) (x_{2i - 1} + x_{2i}). Equation 20 of Flyvbjerg-Petersen
        xi = 0.5 * (xi[0::2] + xi[1::2])

        level += 1

    return pd.DataFrame(levels, columns = ["Level", "N", "Xbar", "c0", "sigma2", "sigma", "dsigma2", "dsigma"])

def plateau(data) -> Optional[int]:
    n = data["N"].to_numpy()
    sigma = data["sigma"].to_numpy()
    dsigma = data["dsigma"].to_numpy()

    # Discard levels which have less then our miminum number of blocks.
    valid = np.where(n >= MINN)[0]
    # If there are less than two of these we have nothing to do.
    if valid.size < 2:
        return None

    # Highest blocking level still having more than miminum number of blocks.
    last = valid[-1]

    # Iterate low blocking to higher blocking and compare consecutive levels
    # for growth of error.
    for i in valid:
        if i + NEXT > last:
            break

        ok = True
        for j in range(i + 1, i + NEXT + 1):
            if abs(sigma[j] - sigma[i]) > (dsigma[j] + dsigma[i]):
                ok = False
                break

            if sigma[i] > 0 and abs(sigma[j] - sigma[i]) / sigma[i] > PLATEAUTOL:
                ok = False
                break

        if ok:
            return int(i)

    # If error keeps growing there is no plateau.
    return None

def main() -> None:
    """
    Extract QMC observables and perform blocking analysis.
    """
    args = parse()

    df = extract(args.path)
    df = prepareObservables(df)

    if args.start is None:
        active = df[
            ~np.isclose(
                df["shift"].to_numpy(dtype = float),
                0.0,
            )
        ]

        start = int(active["iter"].iloc[0])
        print(
            f"Using first active-shift iteration as start: {start}"
        )
    else:
        start = args.start

    df = df[df["iter"] >= start].copy()

    columns = [
        "eproj",
        "ecorr",
        "shift",
        "shiftcorr",
        "nw",
        "nref",
        "nsampled",
        "nsampledo",
    ]

    for column in columns:
        values = df[column].to_numpy(dtype = float)
        values = values[np.isfinite(values)]

        print()
        print(f"{column}:")

        if values.size < 2:
            print("Insufficient finite samples for blocking.")
            continue

        data = blocking(values)

        print(
            data[
                [
                    "Level",
                    "N",
                    "Xbar",
                    "c0",
                    "sigma",
                    "dsigma",
                ]
            ].to_string(index = False)
        )

        level = plateau(data)

        if level is None:
            print("No plateau detected.")
            continue

        row = data.iloc[level]

        print(
            f"Plateau found at level: "
            f"{int(row['Level'])}, "
            f"N: {int(row['N'])}"
        )
        print(f"Xbar: {row['Xbar']}")
        print(f"sigma: {row['sigma']}")
        print(f"dsigma: {row['dsigma']}")

if __name__ == "__main__":
    main()

