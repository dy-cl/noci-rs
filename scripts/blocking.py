# blocking.py

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Plateau picking quantities.
MINN = 8  # At later levels if we have too few blocks the data can become noisy.
NEXT = 1  # Require the next N levels to be consistent within given error bars.
# How much error can change between levels before it is not a plateau.
PLATEAUTOL = 0.25

# Shoulder picking quantities. 
SHOULDER_POINTS = 10
MIN_REFERENCE_POPULATION = 10.0
SHOULDER_TAIL_POINTS = 20
SHOULDER_PROMINENCE = 1.15

# Equilibration picking quantity. 
MSER_START_MAX_FRAC = 0.84


def parse() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Paths to consecutive noci-rs output files.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "First iteration included in the blocking analysis. "
            "If omitted, determine the equilibration start automatically "
            "using MSER after population control begins."
        ),
    )
    return parser.parse_args()


def extract(path: Path) -> pd.DataFrame:
    """Extract the stochastic QMC table from an output file."""

    def isQMCHeader(line: str) -> bool:
        columns = line.split()

        if columns[:6] != [
            "Iter",
            "EProjNum",
            "EProjDen",
            "EProj",
            "ECorr",
            "EShift",
        ]:
            return False

        return columns[6:] in (
            ["NWalk", "NRef", "-", "-"],
            ["NMetric", "NMetricRef", "NSample", "NSampleOcc"],
        )

    floatPattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    optionalPattern = rf"(?:{floatPattern}|-)"

    pattern = re.compile(
        rf"^\s*"
        rf"(\d+)\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({floatPattern})\s+"
        rf"({optionalPattern})\s+"
        rf"({optionalPattern})"
        rf"\s*$"
    )

    rows = []
    inQMC = False
    header = None

    def optionalFloat(value: str) -> float:
        return np.nan if value == "-" else float(value)

    with open(path, "r") as output:
        for line in output:
            if not inQMC:
                if isQMCHeader(line):
                    header = line.split()
                    inQMC = True
                continue

            if line.startswith("===="):
                break

            match = pattern.match(line)

            if match is None:
                continue

            fields = match.groups()

            rows.append(
                (
                    int(fields[0]),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                    float(fields[5]),
                    float(fields[6]),
                    float(fields[7]),
                    optionalFloat(fields[8]),
                    optionalFloat(fields[9]),
                )
            )

    if header is None:
        raise ValueError(f"Stochastic QMC table header not found in {path}")

    df = pd.DataFrame(rows)

    if header[6:] == ["NWalk", "NRef", "-", "-"]:
        df = df.iloc[:, :8]
        df.columns = header[:8]
    else:
        df.columns = header

    return df.drop_duplicates(subset=["Iter"], keep="last")


def prepareObservables(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived observables used in blocking analysis."""
    df = df.copy()

    referenceEnergy = np.nanmedian(df["EProj"] - df["ECorr"])

    shiftActive = ~np.isclose(
        df["EShift"].to_numpy(dtype=float),
        0.0,
    )

    df["EShiftCorr"] = np.where(
        shiftActive,
        df["EShift"] - referenceEnergy,
        np.nan,
    )

    return df


def populationColumns(df: pd.DataFrame):
    """Return total, reference, and occupied population columns."""
    if "NMetric" in df.columns:
        return "NMetric", "NMetricRef", "NSampleOcc"

    if "NWalk" in df.columns:
        return "NWalk", "NRef", None

    raise ValueError("Population columns not found in stochastic QMC table")


def shoulderRows(
    df: pd.DataFrame,
    *,
    npoints=SHOULDER_POINTS,
    minReference=MIN_REFERENCE_POPULATION,
):
    """Return the standard pre-population-control shoulder-estimator rows.
    """
    if df.empty:
        raise ValueError("Empty stochastic QMC trajectory")

    total, reference, _ = populationColumns(df)
    initialShift = float(df["EShift"].iloc[0])

    beforeControl = df[
        np.isclose(
            df["EShift"],
            initialShift,
            rtol=1e-10,
            atol=1e-12,
        )
    ].copy()

    beforeControl = beforeControl[
        beforeControl[reference].abs() >= minReference
    ].copy()

    if len(beforeControl) < npoints:
        raise ValueError(f"Only {len(beforeControl)} eligible pre-control rows")

    beforeControl["Shoulder"] = (
        beforeControl[total] / beforeControl[reference].abs()
    )

    top = beforeControl.nlargest(npoints, "Shoulder").sort_values("Iter")

    return top, beforeControl


def shoulderResolved(
    top: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    prominence=SHOULDER_PROMINENCE,
) -> tuple[bool, float]:
    """Test whether the ratio peak is clearly separated from the late trajectory."""
    if len(top) < SHOULDER_POINTS:
        return False, np.nan

    tailCount = min(SHOULDER_TAIL_POINTS, len(eligible))

    tailRatio = float(eligible["Shoulder"].iloc[-tailCount:].median())
    shoulderRatio = float(top["Shoulder"].mean())

    if not np.isfinite(tailRatio) or tailRatio <= 0.0:
        return True, np.nan

    measuredProminence = shoulderRatio / tailRatio

    return measuredProminence >= prominence, measuredProminence


def standardError(values: pd.Series) -> float:
    """Return the standard error of a finite sample."""
    values = values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.nan

    return float(values.std(ddof=1) / math.sqrt(values.size))


def shoulderSummary(df: pd.DataFrame) -> None:
    """Estimate and print the population shoulder."""
    total, _, occupied = populationColumns(df)

    try:
        top, eligible = shoulderRows(df)
    except ValueError as error:
        print("Shoulder:")
        print(f"No shoulder estimate: {error}")
        return

    resolved, prominence = shoulderResolved(top, eligible)

    print("Shoulder:")

    print()
    print(f"{total}:")
    print(f"Xbar: {top[total].mean()}")
    print(f"sigma: {standardError(top[total])}")

    if occupied is not None and not top[occupied].isna().all():
        print()
        print(f"{occupied}:")
        print(f"Xbar: {top[occupied].mean()}")
        print(f"sigma: {standardError(top[occupied])}")

    print()
    print(
        "Selected iterations: "
        + ",".join(str(int(iteration)) for iteration in top["Iter"])
    )

    if np.isfinite(prominence):
        print(f"Prominence: {prominence}")

    print(f"Resolved: {resolved}")


def blocking(xi) -> pd.DataFrame:
    """Perform recursive Flyvbjerg-Petersen blocking analysis."""
    levels = []

    level = 0
    while xi.size >= 2:
        n = xi.size
        xbar = xi.mean()

        # c_0 = 1 / n \sum_{k = 1}^n (x_k - \bar{x})^2. Equation 8 of
        # Flyvbjerg-Petersen with t = 0.
        c0 = (1 / n) * ((xi - xbar) ** 2).sum()

        # Variance of the sample mean \sigma^2(\bar{x}). Equation 26 of
        # Flyvbjerg-Petersen.
        # \sigma^2(m) = \langle c_0 / (n - 1) \rangle.
        sigma2 = c0 / (n - 1)
        sigma = math.sqrt(sigma2)

        # Error of the above estimator. Equation 28 of Flyvbjerg-Petersen.
        # \sigma^2(m) \approx (c'_0 / (n' - 1)) \pm
        # \sqrt{(2 / (n' - 1))} (c'_0 / (n' - 1)).
        dsigma2 = math.sqrt(2.0 / (n - 1)) * sigma2
        dsigma = sigma / math.sqrt(2.0 * (n - 1))

        levels.append(
            (
                level,
                n,
                xbar,
                c0,
                sigma2,
                sigma,
                dsigma2,
                dsigma,
            )
        )

        # If the data set is odd we must remove 1 element for this to work.
        if n % 2 == 1:
            xi = xi[:-1]
            n -= 1

        # X'_i = (1 / 2) (x_{2i - 1} + x_{2i}). Equation 20 of
        # Flyvbjerg-Petersen.
        xi = 0.5 * (xi[0::2] + xi[1::2])

        level += 1

    return pd.DataFrame(
        levels,
        columns=[
            "Level",
            "N",
            "Xbar",
            "c0",
            "sigma2",
            "sigma",
            "dsigma2",
            "dsigma",
        ],
    )


def blockingRatio(numerator, denominator) -> pd.DataFrame:
    """Block a ratio estimator, including numerator-denominator covariance.

    At every blocking level the projected energy is evaluated as
    ``mean(numerator) / mean(denominator)``. Its uncertainty is obtained by
    first-order propagation of the blocked numerator and denominator means,
    including their covariance.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    if numerator.size != denominator.size:
        raise ValueError(
            "Ratio numerator and denominator have different lengths"
        )

    levels = []
    level = 0

    while numerator.size >= 2:
        n = numerator.size
        numeratorMean = numerator.mean()
        denominatorMean = denominator.mean()

        numeratorDelta = numerator - numeratorMean
        denominatorDelta = denominator - denominatorMean

        numeratorC0 = (numeratorDelta**2).sum() / n
        denominatorC0 = (denominatorDelta**2).sum() / n
        covarianceC0 = (numeratorDelta * denominatorDelta).sum() / n

        numeratorVariance = numeratorC0 / (n - 1)
        denominatorVariance = denominatorC0 / (n - 1)
        covariance = covarianceC0 / (n - 1)

        xbar = numeratorMean / denominatorMean

        sigma2 = (
            numeratorVariance / denominatorMean**2
            + numeratorMean**2
            * denominatorVariance
            / denominatorMean**4
            - 2.0
            * numeratorMean
            * covariance
            / denominatorMean**3
        )

        # Roundoff can make a variance that is analytically non-negative
        # slightly negative when numerator and denominator are very strongly
        # correlated.
        sigma2 = max(float(sigma2), 0.0)
        sigma = math.sqrt(sigma2)
        c0 = sigma2 * (n - 1)
        dsigma2 = math.sqrt(2.0 / (n - 1)) * sigma2
        dsigma = sigma / math.sqrt(2.0 * (n - 1))

        levels.append(
            (
                level,
                n,
                xbar,
                c0,
                sigma2,
                sigma,
                dsigma2,
                dsigma,
            )
        )

        if n % 2 == 1:
            numerator = numerator[:-1]
            denominator = denominator[:-1]

        numerator = 0.5 * (
            numerator[0::2] + numerator[1::2]
        )
        denominator = 0.5 * (
            denominator[0::2] + denominator[1::2]
        )

        level += 1

    return pd.DataFrame(
        levels,
        columns=[
            "Level",
            "N",
            "Xbar",
            "c0",
            "sigma2",
            "sigma",
            "dsigma2",
            "dsigma",
        ],
    )


def plateau(data) -> Optional[int]:
    """Find the first statistically consistent blocking plateau."""
    n = data["N"].to_numpy()
    sigma = data["sigma"].to_numpy()
    dsigma = data["dsigma"].to_numpy()

    # Discard levels which have less than our minimum number of blocks.
    valid = np.where(n >= MINN)[0]

    # If there are less than two of these we have nothing to do.
    if valid.size < 2:
        return None

    # Highest blocking level still having more than minimum number of blocks.
    last = valid[-1]

    # Iterate low blocking to higher blocking and compare consecutive levels
    # for growth of error.
    for i in valid:
        if i + NEXT > last:
            break

        ok = True
        for j in range(i + 1, i + NEXT + 1):
            if abs(sigma[j] - sigma[i]) > (
                dsigma[j] + dsigma[i]
            ):
                ok = False
                break

            if (
                sigma[i] > 0
                and abs(sigma[j] - sigma[i]) / sigma[i]
                > PLATEAUTOL
            ):
                ok = False
                break

        if ok:
            return int(i)

    # If error keeps growing there is no plateau.
    return None


def suffixVariance(values: np.ndarray) -> np.ndarray:
    """Return population variances of every suffix using stable Welford updates."""
    values = np.asarray(values, dtype=float)
    variances = np.empty(values.size, dtype=float)

    count = 0
    mean = 0.0
    m2 = 0.0

    for i in range(values.size - 1, -1, -1):
        count += 1
        delta = values[i] - mean
        mean += delta / count
        delta2 = values[i] - mean
        m2 += delta * delta2
        variances[i] = m2 / count

    return variances


def mserStart(df: pd.DataFrame, column: str) -> int:
    """
    Find an equilibration start using the MSER minimisation criterion.
    """
    finite = np.isfinite(
        df[column].to_numpy(dtype=float)
    )
    data = df.loc[
        finite,
        ["Iter", column],
    ].reset_index(drop=True)
    values = data[column].to_numpy(dtype=float)

    if values.size < 2:
        raise ValueError(
            f"Insufficient {column} samples for MSER analysis"
        )

    variances = suffixVariance(values)
    remaining = np.arange(
        values.size,
        0,
        -1,
        dtype=float,
    )
    mser = variances / remaining

    last = min(
        values.size - 2,
        int(MSER_START_MAX_FRAC * values.size),
    )

    if last < 0:
        raise ValueError(
            f"Insufficient {column} samples for MSER analysis"
        )

    startIndex = int(
        np.nanargmin(mser[: last + 1])
    )

    return int(data["Iter"].iloc[startIndex])


def firstActiveShift(df: pd.DataFrame) -> int:
    """Return the first iteration at which population-control shift is active."""
    active = df[
        ~np.isclose(
            df["EShift"].to_numpy(dtype=float),
            0.0,
        )
    ]

    if active.empty:
        raise ValueError("Shift never becomes active")

    return int(active["Iter"].iloc[0])


def equilibrationStart(df: pd.DataFrame) -> int:
    """Find a conservative automatic equilibration start using MSER."""
    activeStart = firstActiveShift(df)
    active = (
        df[df["Iter"] >= activeStart]
        .copy()
        .reset_index(drop=True)
    )

    energyStart = mserStart(active, "EProj")
    shiftStart = mserStart(active, "EShift")
    start = max(energyStart, shiftStart)

    print(
        f"First active-shift iteration: {activeStart}"
    )
    print(
        f"MSER EProj start: {energyStart}"
    )
    print(
        f"MSER EShift start: {shiftStart}"
    )
    print(
        f"Using equilibration start: {start}"
    )

    return start


def printBlockingResult(data: pd.DataFrame) -> None:
    """Print blocking levels and the first statistically consistent plateau."""
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
        ].to_string(index=False)
    )

    level = plateau(data)

    if level is None:
        print("No plateau detected.")
        return

    row = data.iloc[level]

    print(
        f"Plateau found at level: "
        f"{int(row['Level'])}, "
        f"N: {int(row['N'])}"
    )
    print(f"Xbar: {row['Xbar']}")
    print(f"sigma: {row['sigma']}")
    print(f"dsigma: {row['dsigma']}")


def main() -> None:
    """Extract consecutive QMC outputs and perform shoulder/blocking analysis."""
    args = parse()

    # The files are consecutive sections of the same stochastic trajectory.
    df = pd.concat(
        [extract(path) for path in args.paths],
        ignore_index=True,
    )

    # Restarted output files can contain overlapping iterations. Keep the last
    # occurrence and restore chronological order before analysis.
    df = (
        df.drop_duplicates(
            subset=["Iter"],
            keep="last",
        )
        .sort_values("Iter")
        .reset_index(drop=True)
    )

    df = prepareObservables(df)

    # The shoulder is defined from the constant-shift population-growth region,
    # before the population-controlled data used for blocking analysis.
    shoulderSummary(df)
    print()

    if args.start is None:
        start = equilibrationStart(df)
    else:
        start = args.start
        print(
            f"Using user-specified equilibration start: {start}"
        )

    df = df[
        df["Iter"] >= start
    ].copy()

    if len(df) < 2:
        raise ValueError(
            "Insufficient samples after equilibration start"
        )

    columns = [
        "EProjNum",
        "EProjDen",
        "ECorr",
        "EShift",
        "EShiftCorr",
    ]
    columns.extend(
        column
        for column in [
            "NWalk",
            "NRef",
            "NMetric",
            "NMetricRef",
            "NSample",
            "NSampleOcc",
        ]
        if column in df.columns
    )

    for column in columns:
        if df[column].isna().all():
            continue

        values = df[column].to_numpy(
            dtype=float
        )
        values = values[
            np.isfinite(values)
        ]

        print()
        print(f"{column}:")

        if values.size < 2:
            print(
                "Insufficient finite samples for blocking."
            )
            continue

        printBlockingResult(
            blocking(values)
        )

    # The projected energy is a ratio estimator. Reblock the numerator and
    # denominator together rather than blocking instantaneous EProj values so
    # their covariance contributes to the reported uncertainty.
    finite = (
        np.isfinite(df["EProjNum"])
        & np.isfinite(df["EProjDen"])
    )

    print()
    print("EProj:")

    if finite.sum() < 2:
        print(
            "Insufficient finite samples for blocking."
        )
    else:
        data = blockingRatio(
            df.loc[
                finite,
                "EProjNum",
            ].to_numpy(dtype=float),
            df.loc[
                finite,
                "EProjDen",
            ].to_numpy(dtype=float),
        )
        printBlockingResult(data)


if __name__ == "__main__":
    main()
