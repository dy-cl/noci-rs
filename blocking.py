import re
from typing import Optional
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import math

# Plateau picking quantities.
MINN = 16 # At later levels if we have too few blocks the data can become noist.
NEXT = 3 # Require the next N levels to be consistent within given error bars.
PLATEAUTOL = 0.25 # How much error can change between levels before it is not a plateau.

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("path", type = Path)
    p.add_argument("--start", type = int)
    return p.parse_args()

def extract(path) -> pd.DataFrame:
    floatre = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    pattern = (
        r"^\s*(\d+)\s+"     # iter
        + floatre + r"\s+"  # E (eproj)
        + floatre + r"\s+"  # Ecorr
        + floatre + r"\s+"  # Es
        + floatre + r"\s+"  # EsS
        + floatre + r"\s+"  # Nw (||C||)
        + floatre + r"\s+"  # Nw (||SC||)
        + floatre + r"\s*$" # Nref (||SC||)
    )

    rows = []
    with open(path, "r") as f:
        for line in f:
            m = re.match(pattern, line)
            if not m:
                continue
            it = int(m.group(1))
            eproj = float(m.group(2))
            ecorr = float(m.group(3))
            es = float(m.group(4))
            es_s = float(m.group(5))
            nwc = float(m.group(6))
            nwsc = float(m.group(7))
            nrefsc = float(m.group(8))
            rows.append((it, eproj, ecorr, es, es_s, nwc, nwsc, nrefsc))

    df = pd.DataFrame(rows, columns = ["iter", "eproj", "ecorr", "es", "esS", "nwc", "nwsc", "nrefsc"])
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
    for i in valid[:-1]:
        if i + NEXT > last:
            break
        
        ok = True
        j = i + 1
        if abs(sigma[j] - sigma[i]) <= (dsigma[j] + dsigma[i]):
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
    args = parse()
    df = extract(args.path)

    df = df[df["iter"] >= args.start]
    cols = ["eproj", "ecorr", "es", "esS", "nwc", "nwsc", "nrefsc"]

    for col in cols:
        print(" ")

        x = df[col].to_numpy(dtype = float)
        data = blocking(x)
        print(f"{col}:")
        print(data[["Level", "N", "Xbar", "c0", "sigma", "dsigma"]].to_string(index = False))
        
        # Extract plateau'd row and print.
        k = plateau(data) 
        if k is None:
            print("No plateau detected.")
        else:
            row = data.iloc[k]
            print(f"Plateau found at level: {row['Level']}, N: {row['N']}")
            print(f"Xbar: {row['Xbar']}")
            print(f"sigma: {row['sigma']}")
            print(f"dsigma: {row['dsigma']}")

if __name__ == "__main__":
    main()

