# plotting.py

from pathlib import Path
import argparse
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Plotting parameters

TICKFONTSIZE = 32
LABELFONTSIZE = 42
LINEWIDTH = 5
MARKERSIZE = 10

# All required regex goes here.

ENERGYRREGEX = re.compile(r"^\s*R:\s*([+-]?\d+(?:\.\d+)?)", re.MULTILINE)
ENERGYSTATEREGEX = re.compile(
    r"^\s*State\((?P<idx>[^)]+)\):\s*"
    r"(?:(?P<label>.*?)\s*(?:,\s*)?)?"
    r"E\s*[:=]\s*(?P<E>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
    re.MULTILINE,
)

DETITERREGEX = re.compile(r"iter\s+(\d+)", re.IGNORECASE)
DETRELEVANTHEADERREGEX = re.compile(r"^Relevant space coefficients")
DETNULLHEADERREGEX = re.compile(r"^Null space coefficients")
DETFULLHEADERREGEX = re.compile(r"^Full coefficients")
DETCOEFFLINEREGEX = re.compile(
    r"^\s*(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

QMCFLOATREGEX = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
QMCLINEREGEX = re.compile(
    r"^\s*(\d+)\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s+"
    + QMCFLOATREGEX + r"\s*"
    + r"(\d+)\s*$"
)

def readQMC(path: Path) -> pd.DataFrame:
    """
    Read in QMC lines from a NOCIQMC output file and create a pandas dataframe.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            m = QMCLINEREGEX.match(line)
            if not m:
                continue
            rows.append((
                int(m.group(1)),
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
                float(m.group(6)),
                float(m.group(7)),
                float(m.group(8)),
                float(m.group(9)),
                int(m.group(10)),
            ))
    return pd.DataFrame(rows, columns = ["iter", "eproj", "ecorr", "es", "ess", "nwc", "nrefc", "nwsc", "nrefsc", "nocc"])


def readDeterministicCoefficients(path: Path) -> pd.DataFrame:
    """
    Read in deterministic coefficient lines from a deterministic NOCIQMC output file and create a pandas dataframe.
    """
    rows = []
    currentIter = None
    section = None

    with open(path, "r") as f:
        for line in f:
            s = line.strip()

            mIter = DETITERREGEX.match(s)
            if mIter:
                currentIter = int(mIter.group(1))
                section = None
                continue

            if DETRELEVANTHEADERREGEX.match(s):
                section = "relevant"
                continue
            if DETNULLHEADERREGEX.match(s):
                section = "null"
                continue
            if DETFULLHEADERREGEX.match(s):
                section = None
                continue

            mCoeff = DETCOEFFLINEREGEX.match(s)
            if mCoeff is None or currentIter is None or section is None:
                continue

            state = int(mCoeff.group(1))
            coeff = float(mCoeff.group(2))
            rows.append((currentIter, section, state, coeff))

    return pd.DataFrame(rows, columns = ["iter", "space", "state", "coeff"])

def shiftChange(series: pd.Series):
    """
    Find the first iteration where either of the shifts change. Or more generally any series.
    """
    x0 = series.iloc[0]
    mask = ~np.isclose(series.to_numpy(), x0)
    if mask.any():
        return int(series.index[mask.argmax()]), int(series.iloc[mask.argmax()])
    return None, None

def readEnergy(path: Path) -> pd.DataFrame:
    """
    Read in energy values for calculations performed across a range of geometries and create a dataframe.
    """
    text = path.read_text()

    rPositions = [(m.start(), float(m.group(1))) for m in ENERGYRREGEX.finditer(text)]
    rPositions.append((len(text), None))

    rows = []
    for i in range(len(rPositions) - 1):
        start, R = rPositions[i]
        end, _ = rPositions[i + 1]
        if R is None:
            continue

        block = text[start:end]
        for m in ENERGYSTATEREGEX.finditer(block):
            idx = m.group("idx")
            rawLabel = (m.group("label") or "").strip()
            E = float(m.group("E"))
            label = createEnergyLabel(idx, rawLabel)
            rows.append((R, idx, rawLabel, label, E))

    return pd.DataFrame(rows, columns = ["R", "idx", "rawLabel", "label", "E"]).sort_values(["label", "R"])

def createEnergyLabel(idx: str, rawLabel: str) -> str:
    """
    Convert state labels in `noci-rs` to desired form for plotting.
    """
    if idx.startswith("NOCI-qmc"):
        return "NOCIQMC"
    if idx.startswith("NOCI"):
        return "NOCI"
    if idx.startswith("FCI"):
        return "FCI"
    return rawLabel if rawLabel else f"State({idx})"

def readMatrix(path: Path) -> pd.DataFrame:
    """
    Read in a matrix and create a dataframe.
    """
    M = np.loadtxt(path)
    return pd.DataFrame(M)

def readHistogram(filepath: Path) -> pd.DataFrame:
    with open(filepath, "r") as f:
        logmin, logmax, nbins = f.readline().split()
        logmin = float(logmin)
        logmax = float(logmax)
        nbins = int(nbins)

        ntotal, nlow, nhigh = map(int, f.readline().split())
        counts = np.array([int(f.readline()) for _ in range(nbins)])

    binedges = np.linspace(logmin, logmax, nbins + 1)
    bincenters = 0.5 * (binedges[:-1] + binedges[1:])
    binwidth = binedges[1] - binedges[0]

    return pd.DataFrame({"file": filepath.name, "logmin": logmin, "logmax": logmax, "nbins": nbins, "ntotal": ntotal,
                         "nlow": nlow, "nhigh": nhigh, "bin": np.arange(nbins), "binLeft": binedges[:-1], "binRight": binedges[1:],
                         "binCenter": bincenters, "binWidth": binwidth, "count": counts})

def formatAxes(xlabel = None, ylabel = None, legend = False, legendLoc = None):
    """
    Apply uniform axis formatting.
    """
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = LABELFONTSIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = LABELFONTSIZE)
    plt.xticks(fontsize = TICKFONTSIZE)
    plt.yticks(fontsize = TICKFONTSIZE)
    if legend:
        if legendLoc is None:
            plt.legend(fontsize = LABELFONTSIZE)
        else:
            plt.legend(fontsize = LABELFONTSIZE, loc = legendLoc)

def plotDeterministicCoefficients(args):
    """
    Plot deterministic relevant or null-space coefficients.
    """
    df = readDeterministicCoefficients(args.path)
    df = df[df["space"] == args.space].copy()

    finalIter = df["iter"].max()
    finalDf = df[df["iter"] == finalIter].copy()
    finalDf["absCoeff"] = finalDf["coeff"].abs()

    ntop = min(args.ncoeffs, len(finalDf))
    topStates = finalDf.nlargest(ntop, "absCoeff")["state"].to_numpy()

    ylabel = (
        r"$\langle \Psi_{\Lambda} | \hat P_r | \Psi(\tau)\rangle$"
        if args.space == "relevant"
        else r"$\langle \Psi_{\Lambda} | \hat P_n | \Psi(\tau)\rangle$"
    )

    setStyle()
    plt.figure()
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, ntop))

    for color, state in zip(colors, topStates):
        g = df[df["state"] == state].sort_values("iter")
        if args.space == "relevant":
            plt.plot(g["iter"], g["coeff"], linewidth = LINEWIDTH, color = color)
        else:
            plt.plot(g["iter"], g["coeff"], linewidth = LINEWIDTH, color = "tab:grey")

    formatAxes(xlabel = "Iteration", ylabel = ylabel)
    plt.subplots_adjust(left = 0.15)
    finish(args)


def plotEnergy(args):
    """
    Plot energies across a geometry scan.
    """
    df = readEnergy(args.path)

    setStyle()
    fig, ax = plt.subplots()

    def plotLabel(lbl, display = None, **kwargs):
        g = df[df["label"] == lbl].sort_values("R")
        if not g.empty:
            ax.plot(g["R"], g["E"], label = display if display is not None else lbl, **kwargs)

    plotLabel("NOCI", display = r"$|\Psi^{\mathrm{NOCI}}\rangle$", linewidth = LINEWIDTH, zorder = 10, color = "tab:green")
    plotLabel("NOCIQMC", display = r"$|\Psi^{\mathrm{NOCI\!-\!QMC}}\rangle$", linewidth = LINEWIDTH, linestyle = "--", zorder = 20, color = "tab:pink")

    gFCI = df[df["label"] == "FCI"].sort_values("R")
    if not gFCI.empty:
        ax.plot(gFCI["R"], gFCI["E"], label = "FCI", color = "black", marker = "o", linestyle = " ", markersize = MARKERSIZE, zorder = 30)

    seen = set()
    autoLabels = sorted(l for l in df["label"].unique() if l.startswith("RHF") or l.startswith("UHF") or l.startswith("M "))
    for lbl in autoLabels:
        g = df[df["label"] == lbl].sort_values("R")
        if g.empty:
            continue

        if "RHF" in lbl:
            color = "tab:red"
            display = r"$|\Psi^{\mathrm{RHF}}\rangle$"
        elif "UHF" in lbl:
            color = "tab:blue"
            display = r"$|\Psi^{\mathrm{UHF}}\rangle$"
        else:
            color = None
            display = lbl

        label = display if display not in seen else None
        seen.add(display)
        ax.plot(g["R"], g["E"], linewidth = LINEWIDTH, color = color, label = label)

    formatAxes(xlabel = "R / Å", ylabel = "E / Ha", legend = True)
    plt.grid(True)
    fig.subplots_adjust(left = 0.14)
    finish(args)


def plotMatrix(args):
    """
    Plot a matrix heatmap.
    """
    df = readMatrix(args.path)
    M = df.to_numpy()

    vmin = 0.0
    vmax = float(np.max(M)) if np.max(M) > 0 else 1.0

    plt.figure()
    im = plt.imshow(M, cmap = "inferno", vmin = vmin, vmax = vmax, interpolation = "nearest")
    plt.colorbar(im)
    plt.xticks(fontsize = TICKFONTSIZE)
    plt.yticks(fontsize = TICKFONTSIZE)
    plt.tight_layout()
    finish(args)


def plotExcitationHist(args):
    """
    Plot the combined excitation histogram.
    """
    df = readExcitationHistogram(args.path)

    setStyle()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: rf"$10^{{{int(np.round(x))}}}$"))

    norm = mcolors.Normalize(vmin = df["binCenter"].min(), vmax = df["binCenter"].max())
    cmap = plt.cm.winter
    colors = cmap(norm(df["binCenter"].to_numpy()))

    ax.bar(df["binCenter"], df["count"], width = df["binWidth"], align = "center", color = colors)
    formatAxes(
        xlabel = r"$\frac{\Delta\tau|H_{\Pi\Omega} - E_s^S(\tau)S_{\Pi\Omega}|}{P_{\text{gen}(\Pi|\Omega)}}$",
        ylabel = "Frequency"
    )
    finish(args)


def plotNW(args):
    """
    Plot walker populations against iteration.
    """
    df = readQMC(args.path)

    _, iterEs = shiftChange(df["es"])
    _, iterEsS = shiftChange(df["ess"])

    setStyle()
    plt.figure()
    plt.plot(df["iter"], df["nwc"], label = r"$N_w(\tau)$", linewidth = LINEWIDTH)
    plt.plot(df["iter"], df["nwsc"], label = r"$\tilde{N}_w(\tau)$", linewidth = LINEWIDTH)
    if iterEs is not None:
        plt.axvline(iterEs, linestyle = "--", linewidth = LINEWIDTH, color = "tab:blue")
    if iterEsS is not None:
        plt.axvline(iterEsS, linestyle = "--", linewidth = LINEWIDTH, color = "tab:orange")
    formatAxes(xlabel = r"Iteration / $\tau$", ylabel = r"$N_w(\tau)$", legend = True)
    finish(args)


def plotProjectedShift(args):
    """
    Plot projected and shift energies against iteration.
    """
    df = readQMC(args.path)

    _, iterEs = shiftChange(df["es"])
    _, iterEsS = shiftChange(df["ess"])

    setStyle()
    plt.figure()
    plt.plot(df["iter"], df["es"], label = r"$E_s(\tau)$", linewidth = LINEWIDTH, color = "tab:blue")
    plt.plot(df["iter"], df["ess"], label = r"$E_s^S(\tau)$", linewidth = LINEWIDTH, color = "tab:orange")
    if iterEs is not None:
        plt.axvline(iterEs, linestyle = "--", linewidth = LINEWIDTH, color = "tab:blue")
    if iterEsS is not None:
        plt.axvline(iterEsS, linestyle = "--", linewidth = LINEWIDTH, color = "tab:orange")
    plt.plot(df["iter"], df["ecorr"], label = r"$E_{\mathrm{Proj}}(\tau)$", linewidth = LINEWIDTH, color = "tab:green")
    formatAxes(xlabel = r"Iteration / $\tau$", ylabel = "Energy / Ha", legend = True, legendLoc = "lower right")
    finish(args)


def plotShoulder(args):
    """
    Plot total to reference ratios against walker number.
    """
    df = readQMC(args.path)

    ratioC = df["nwc"] / df["nrefc"]
    ratioSC = df["nwsc"] / df["nrefsc"]

    setStyle()
    plt.figure()
    plt.plot(df["nwsc"], ratioSC, linewidth = LINEWIDTH, color = "tab:orange", label = r"$\frac{\tilde N_w(\tau)}{\tilde N_{w,\mathrm{ref}}}$")
    plt.plot(df["nwc"], ratioC, linewidth = LINEWIDTH, color = "tab:blue", label = r"$\frac{N_w(\tau)}{N_{w,\mathrm{ref}}}$")
    formatAxes(legend = True)
    finish(args)

def setStyle():
    """
    Choose a uniform font for all plotting purposes.
    """
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

def addCommonArgs(parser):
    """
    Add common output arguments to a subparser.
    """
    parser.add_argument("--save", type = Path, default = None)
    parser.add_argument("--dpi", type = int, default = 300)

def buildParser():
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = "plot", required = True)

    p = subparsers.add_parser("deterministic-coefficients")
    p.add_argument("path", type = Path)
    p.add_argument("--ncoeffs", type = int, default = 10)
    p.add_argument("--space", choices = ["relevant", "null"], default = "relevant")
    addCommonArgs(p)
    p.set_defaults(func = plotDeterministicCoefficients)

    p = subparsers.add_parser("energy")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotEnergy)

    p = subparsers.add_parser("matrix")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotMatrix)

    p = subparsers.add_parser("excitation-hist")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotExcitationHist)

    p = subparsers.add_parser("nw")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotNW)

    p = subparsers.add_parser("projected-shift")
    p.add_argument("path", type = Path)
    p.add_argument("--propagator", "-p", type = str, default = None)
    addCommonArgs(p)
    p.set_defaults(func = plotProjectedShift)

    p = subparsers.add_parser("shoulder")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotShoulder)

    return parser

def finish(args):
    """
    Save a figure or show it interactively.
    """
    if args.save is not None:
        plt.savefig(args.save, dpi = args.dpi, bbox_inches = "tight")
    else:
        plt.show()

def main():
    """
    Parse arguments and dispatch the selected plot.
    """
    args = buildParser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
