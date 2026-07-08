# plotting.py

from pathlib import Path
import argparse
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
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
    r"^\s*(?P<hprefix>h-)?State\((?P<idx>[^)]+)\):\s*"
    r"(?:(?P<label>.*?)\s*(?:,\s*)?)?"
    r"E\s*[:=]\s*(?P<E>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
    r"(?:\s*[+-]\s*[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?i)?",
    re.MULTILINE,
)
DETITERREGEX = re.compile(r"iter\s+(\d+)", re.IGNORECASE)
DETRELEVANTHEADERREGEX = re.compile(r"^Relevant space coefficients")
DETNULLHEADERREGEX = re.compile(r"^Null space coefficients")
DETFULLHEADERREGEX = re.compile(r"^Full coefficients")
DETCOEFFLINEREGEX = re.compile(
    r"^\s*(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
ISHIFTREGEX = re.compile(
    r"(?:iShift\s*=|i\s*:)\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)

LIVE_READ_ERRORS = (
    OSError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)


def addTrajectoryArgs(parser):
    """
    Add arguments for trajectory plots, including interactive live updating.
    """
    addCommonArgs(parser)
    parser.add_argument(
        "--live",
        action = "store_true",
        help = "Update the interactive Matplotlib window as the output file grows.",
    )
    parser.add_argument(
        "--interval",
        type = float,
        default = 2.0,
        help = "Seconds between live plot updates.",
    )


def showLive(args, fig, update):
    """
    Show a Matplotlib window and update it while it remains open.
    """
    if args.save is not None:
        raise ValueError("--live cannot be combined with --save")

    plt.show(block = False)

    while plt.fignum_exists(fig.number):
        try:
            update()
        except LIVE_READ_ERRORS:
            # The file may temporarily contain a partially written final line.
            pass

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(args.interval)

def findQMC(path: Path):
    """
    Find where QMC starts.
    """
    start = None
    nrows = 0
    inqmc = False

    with open(path, "r") as f:
        for lineno, line in enumerate(f):
            if not inqmc:
                header = line.lstrip()

                if (
                    header.startswith("Iter")
                    and "EProjNum" in header
                    and "EProjDen" in header
                    and "EProj" in header
                    and "ECorr" in header
                    and "EShift" in header
                    and "NW" in header
                    and "NRef" in header
                    and "NSample" in header
                    and "NDet (Sampled)" in header
                ):
                    start = lineno + 1
                    inqmc = True

                continue

            if line.startswith("===="):
                break

            s = line.lstrip()
            if not s:
                continue

            if s[0].isdigit():
                nrows += 1

    return start, nrows

def readQMC(path: Path) -> pd.DataFrame:
    """
    Read the stochastic QMC table from an output file.
    """
    start, nrows = findQMC(path)

    return pd.read_csv(
        path,
        sep = r"\s+",
        header = None,
        skiprows = start,
        nrows = nrows,
        names = [
            "Iter",
            "EProjNum",
            "EProjDen",
            "EProj",
            "ECorr",
            "EShift",
            "NW",
            "NRef",
            "NSample",
            "NDet (Sampled)",
        ],
        engine = "c",
    )

def findDeterministicQMC(path: Path):
    """
    Find where the deterministic propagation table starts.
    """
    start = None
    nrows = 0
    inQMC = False

    with open(path, "r") as f:
        for lineno, line in enumerate(f):
            if not inQMC:
                if (
                    line.lstrip().startswith("iter")
                    and "|dE|" in line
                    and "Shift (Es)" in line
                    and "Shift (EsS)" in line
                    and "||C||" in line
                    and "||SC||" in line
                ):
                    start = lineno + 1
                    inQMC = True
                continue

            if line.startswith("===="):
                break

            s = line.lstrip()
            if not s:
                continue

            if s[0].isdigit():
                nrows += 1

    if start is None:
        raise ValueError("Deterministic propagation table header not found")

    return start, nrows


def readDeterministicQMC(path: Path) -> pd.DataFrame:
    """
    Read the deterministic propagation table from an output file.
    """
    start, nrows = findDeterministicQMC(path)

    return pd.read_csv(
        path,
        sep = r"\s+",
        header = None,
        skiprows = start,
        nrows = nrows,
        names = [
            "iter",
            "energy",
            "de",
            "es",
            "ess",
            "nwc",
            "nwsc",
            "metric_norm",
        ],
        engine = "c",
    )

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
    arr = np.asarray(series)
    x0 = arr[0]
    mask = ~np.isclose(arr, x0)
    if mask.any():
        idx = int(mask.argmax())
        return idx, arr[idx]
    return None, None

def qmcShiftCorrelation(df: pd.DataFrame):
    """
    Convert the total QMC shift to a correlation-energy shift.

    Before population control begins, the printed shift is zero. Those
    entries are returned as NaN so they are not drawn as physical shifts.
    """
    iterShift, _ = shiftChange(df["EShift"])

    shiftCorr = pd.Series(
        np.nan,
        index = df.index,
        dtype = float,
    )

    if iterShift is None:
        return shiftCorr, None

    referenceEnergy = np.nanmedian(
        df["EProj"] - df["ECorr"]
    )

    shiftCorr.iloc[iterShift:] = (
        df["EShift"].iloc[iterShift:]
        - referenceEnergy
    )

    return shiftCorr, iterShift

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
            if m.group("hprefix") and rawLabel and not rawLabel.startswith("h-"):
                if rawLabel.startswith("RHF") or rawLabel.startswith("UHF"):
                    rawLabel = f"h-{rawLabel}"
            E = float(m.group("E"))
            label = createEnergyLabel(idx, rawLabel)
            rows.append((R, idx, rawLabel, label, E))

    return pd.DataFrame(rows, columns = ["R", "idx", "rawLabel", "label", "E"]).sort_values(["label", "R"])

def readEnergyTable(path: Path, label = None) -> pd.DataFrame:
    """
    Read a simple energy table with columns E and R.
    The column order shoule be R E`.
    """
    df = pd.read_csv(path, sep = r"\s+|,", engine = "python", comment = "#")
    cols = {c.strip().lower(): c for c in df.columns}

    if "e" not in cols or "r" not in cols:
        raise ValueError(f"{path} must contain columns named E and R")

    out = pd.DataFrame({
        "R": pd.to_numeric(df[cols["r"]]),
        "E": pd.to_numeric(df[cols["e"]]),
    })
    out["label"] = label if label is not None else path.stem
    return out.sort_values("R")

def createEnergyLabel(idx: str, rawLabel: str) -> str:
    """
    Convert state labels in `noci-rs` to desired form for plotting.
    """
    if idx.startswith("NOCI-qmc"):
        return "NOCIQMC"
    if idx.startswith("NOCI-PT2"):
        m = ISHIFTREGEX.search(idx)
        if m is not None:
            shift = float(m.group(1))
            return f"NOCI-PT2 iShift={shift:g}"
        return "NOCI-PT2"
    #if idx.startswith("SNOCI"):
    #    return "SNOCI"
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
    Legends are always placed outside the plotting axes for consistency.
    """
    ax = plt.gca()
    fig = ax.figure

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = LABELFONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize = LABELFONTSIZE)

    ax.tick_params(axis = "both", labelsize = TICKFONTSIZE)

    if legend:
        ax.legend(
            fontsize = LABELFONTSIZE,
            loc = "center left",
            bbox_to_anchor = (1.02, 0.5),
            frameon = False,
            borderaxespad = 0.0,
        )
    fig.subplots_adjust(right = 0.75)

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

def addEnergyInset(ax, xlim, ylim, loc = [0.64, 0.22, 0.25, 0.25], side = "right"):
    """
    Add an inset showing a zoomed region of the main energy plot.

    `loc` is [left, bottom, width, height] in parent-axes fraction coordinates.
    """
    inset = ax.inset_axes(loc)

    for line in ax.get_lines():
        inset.plot(
            line.get_xdata(),
            line.get_ydata(),
            linewidth = max(1.5, 0.55 * line.get_linewidth()),
            linestyle = line.get_linestyle(),
            marker = line.get_marker(),
            markersize = max(3, 0.55 * line.get_markersize()),
            color = line.get_color(),
            zorder = line.get_zorder(),
        )

    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.tick_params(axis = "both", labelsize = 16)
    inset.grid(True)

    xmin, xmax = xlim
    ymin, ymax = ylim

    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill = False,
        edgecolor = "0.35",
        linewidth = 2,
        linestyle = "--",
        alpha = 0.5,
        zorder = 50,
    )
    ax.add_patch(rect)

    if side == "right":
        connectors = [
            ((xmax, ymax), (0.0, 1.0)),
            ((xmax, ymin), (0.0, 0.0)),
        ]
    elif side == "left":
        connectors = [
            ((xmin, ymax), (1.0, 1.0)),
            ((xmin, ymin), (1.0, 0.0)),
        ]
    elif side == "above":
        connectors = [
            ((xmin, ymax), (0.0, 0.0)),
            ((xmax, ymax), (1.0, 0.0)),
        ]
    else:
        connectors = [
            ((xmin, ymin), (0.0, 1.0)),
            ((xmax, ymin), (1.0, 1.0)),
        ]

    for xyMain, xyInset in connectors:
        con = ConnectionPatch(
            xyA = xyMain,
            coordsA = ax.transData,
            xyB = xyInset,
            coordsB = inset.transAxes,
            color = "0.35",
            linewidth = 2,
            linestyle = "--",
            alpha = 0.5,
            zorder = 60,
            clip_on = False,
        )
        ax.add_artist(con)

    return inset 

def plotEnergy(args):
    """
    Plot energies across a geometry scan.
    """
    df = readEnergy(args.path)

    tablePaths = list(args.tables_pos) + list(args.tables_opt)
    tableDfs = []
    for i, path in enumerate(tablePaths):
        label = args.table_label[i] if i < len(args.table_label) else path.stem
        tableDfs.append(readEnergyTable(path, label = label))

    setStyle()
    fig, ax = plt.subplots()

    def plotLabel(lbl, display = None, **kwargs):
        g = df[df["label"] == lbl].sort_values("R")
        if not g.empty:
            ax.plot(g["R"], g["E"], label = display if display is not None else lbl, **kwargs)
    
    plotLabel("NOCI", display = r"$|\Psi^{\mathrm{NOCI}}\rangle$", linewidth = LINEWIDTH, zorder = 10, color = "tab:green")
    
    pt2Labels = sorted(
        [lbl for lbl in df["label"].unique() if lbl.startswith("NOCI-PT2")],
        key = lambda lbl: float(ISHIFTREGEX.search(lbl).group(1)) if ISHIFTREGEX.search(lbl) else -1.0,
    )

    pt2Shifts = [
        float(ISHIFTREGEX.search(lbl).group(1))
        for lbl in pt2Labels
        if ISHIFTREGEX.search(lbl)
    ]

    if pt2Labels:
        baseCmap = plt.get_cmap("Purples")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "Purples_trunc",
            baseCmap(np.linspace(0.5, 0.9, 256)),
        )
        if pt2Shifts:
            norm = mcolors.Normalize(vmin = min(pt2Shifts), vmax = max(pt2Shifts))
        else:
            norm = mcolors.Normalize(vmin = 0.0, vmax = 1.0)

        for i, lbl in enumerate(pt2Labels):
            m = ISHIFTREGEX.search(lbl)
            shift = float(m.group(1)) if m else 0.0
            color = cmap(norm(shift)) if pt2Shifts else "tab:purple"

            plotLabel(
                lbl,
                display = r"$|\Psi^{\mathrm{NOCI-PT2}}\rangle$" if i == len(pt2Labels) - 1 else "_nolegend_",
                linewidth = LINEWIDTH,
                linestyle = "-",
                zorder = 15,
                color = color,
            )

        if len(set(pt2Shifts)) > 1:
            sm = plt.cm.ScalarMappable(norm = norm, cmap = cmap)
            sm.set_array([])
            
            cax = ax.inset_axes([0.08, 0.90, 0.28, 0.035])
            cbar = fig.colorbar(sm, cax = cax, orientation = "horizontal")
            cbar.set_label(r"$\epsilon / E_h$", fontsize = 20)
            cbar.ax.tick_params(labelsize = 16)
    
    plotLabel("SNOCI", display = r"$|\Psi^{\mathrm{SNOCI}}\rangle$", linewidth = LINEWIDTH, linestyle = ":", zorder = 16, color = "tab:cyan")
    plotLabel("NOCIQMC", display = r"$|\Psi^{\mathrm{NOCI\!-\!QMC}}\rangle$", linewidth = LINEWIDTH, zorder = 20, color = "tab:pink")

    gFCI = df[df["label"] == "FCI"].sort_values("R")
    if not gFCI.empty:
        ax.plot(gFCI["R"], gFCI["E"], label = "FCI", color = "black", marker = "o", linestyle = " ", markersize = MARKERSIZE, zorder = 30)

    seen = set()
    autoLabels = sorted(l for l in df["label"].unique() if l.startswith("RHF") or l.startswith("UHF") or l.startswith("h-RHF") or l.startswith("h-UHF") or l.startswith("M "))
    for lbl in autoLabels:
        g = df[df["label"] == lbl].sort_values("R")
        if g.empty:
            continue

        linestyle = "-"
        if lbl.startswith("h-RHF"):
            color = "tab:red"
            display = r"$|\tilde{\Psi}^{\mathrm{h\!-\!RHF}}\rangle$"
            linestyle = "--"
        elif lbl.startswith("h-UHF"):
            color = "tab:blue"
            display = r"$|\tilde{\Psi}^{\mathrm{h\!-\!UHF}}\rangle$"
            linestyle = "--"
        elif "RHF" in lbl:
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

        gPlot = g
        if lbl.startswith("h-UHF") or lbl.startswith("h-RHF"):
            parentLbl = lbl.removeprefix("h-")
            gParent = df[df["label"] == parentLbl].sort_values("R")
            if not gParent.empty:
                gPlot = pd.concat([g, gParent.head(1)], ignore_index = True).sort_values("R")

        ax.plot(gPlot["R"], gPlot["E"], linewidth = LINEWIDTH, color = color, linestyle = linestyle, label = label)

    for tableDf in tableDfs:
        rawLabel = tableDf["label"].iloc[0]
        display = rf"$|\Psi^{{\mathrm{{{rawLabel}}}}}\rangle$"

        ax.plot(
            tableDf["R"],
            tableDf["E"],
            label = display,
            linewidth = LINEWIDTH,
            linestyle = "-",
            zorder = 1,
            color = 'tab:orange'
        )

    formatAxes(xlabel = "R / Å", ylabel = "E / Ha", legend = True)
    ax.grid(True)

    if args.inset is not None:
        xmin, xmax, ymin, ymax = args.inset
        left, bottom = args.inset_location
        width, height = args.inset_size

        addEnergyInset(
            ax,
            xlim = (xmin, xmax),
            ylim = (ymin, ymax),
            loc = [left, bottom, width, height],
            side = args.inset_side
        )

    formatAxes(xlabel = "R / Å", ylabel = "E / Ha", legend = True)
    plt.grid(True)
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

def plotProjectedShift(args):
    """
    Plot projected and population-control shift correlation energies.
    """
    if not args.live:
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EProj",
                "ECorr",
                "EShift",
            ]
        )

        shiftCorr, iterShift = qmcShiftCorrelation(df)

        setStyle()
        plt.figure()


        if iterShift is not None:
            plt.plot(
                df["Iter"],
                shiftCorr,
                label = r"$E_s^S(\tau)$",
                linewidth = LINEWIDTH,
                color = "tab:blue",
            )

            plt.axvline(
                df["Iter"].iloc[iterShift],
                linestyle = "--",
                linewidth = LINEWIDTH,
                color = "tab:blue",
            )

        plt.plot(
            df["Iter"],
            df["ECorr"],
            label = r"$E_{\mathrm{Proj}}(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:green",
        )

        formatAxes(
            xlabel = r"Iteration / $\tau$",
            ylabel = "Energy / Ha",
            legend = True,
            legendLoc = "lower right",
        )

        finish(args)
        return

    setStyle()
    fig, ax = plt.subplots()

    lineShift, = ax.plot(
        [],
        [],
        label = r"$E_s^S(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:blue",
    )

    lineEproj, = ax.plot(
        [],
        [],
        label = r"$E_{\mathrm{Proj}}(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:green",
    )

    shiftLine = ax.axvline(
        0.0,
        linestyle = "--",
        linewidth = LINEWIDTH,
        color = "tab:blue",
        visible = False,
    )

    formatAxes(
        xlabel = r"Iteration / $\tau$",
        ylabel = "Energy / Ha",
        legend = True,
        legendLoc = "lower right",
    )

    def update():
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EProj",
                "ECorr",
                "EShift",
            ]
        )

        if df.empty:
            return

        x = df["Iter"].to_numpy()
        shiftCorr, iterShift = qmcShiftCorrelation(df)

        lineEproj.set_data(
            x,
            df["ECorr"].to_numpy(),
        )

        lineShift.set_data(
            x,
            shiftCorr.to_numpy(),
        )

        if iterShift is not None:
            value = df["Iter"].iloc[iterShift]
            shiftLine.set_xdata([value, value])
            shiftLine.set_visible(True)
        else:
            shiftLine.set_visible(False)

        ax.relim()
        ax.autoscale_view()

    showLive(args, fig, update)

def plotNW(args):
    """
    Plot persistent and sampled-source populations against iteration.
    """
    if not args.live:
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EShift",
                "NW",
                "NSample",
            ]
        )

        iterShift, _ = shiftChange(df["EShift"])

        setStyle()
        plt.figure()

        plt.plot(
            df["Iter"],
            df["NW"],
            label = r"$N_w(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:blue",
        )

        plt.plot(
            df["Iter"],
            df["NSample"],
            label = r"$N_{\mathrm{sampled}}(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:orange",
        )

        if iterShift is not None:
            plt.axvline(
                df["Iter"].iloc[iterShift],
                linestyle = "--",
                linewidth = LINEWIDTH,
                color = "tab:blue",
            )

        formatAxes(
            xlabel = r"Iteration / $\tau$",
            ylabel = "Population",
            legend = True,
        )

        finish(args)
        return

    setStyle()
    fig, ax = plt.subplots()

    lineNw, = ax.plot(
        [],
        [],
        label = r"$N_w(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:blue",
    )

    lineSampled, = ax.plot(
        [],
        [],
        label = r"$N_{\mathrm{sampled}}(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:orange",
    )

    shiftLine = ax.axvline(
        0.0,
        linestyle = "--",
        linewidth = LINEWIDTH,
        color = "tab:blue",
        visible = False,
    )

    formatAxes(
        xlabel = r"Iteration / $\tau$",
        ylabel = "Population",
        legend = True,
    )

    def update():
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EShift",
                "NW",
                "NSample",
            ]
        )

        if df.empty:
            return

        x = df["Iter"].to_numpy()

        lineNw.set_data(
            x,
            df["NW"].to_numpy(),
        )

        lineSampled.set_data(
            x,
            df["NSample"].to_numpy(),
        )

        iterShift, _ = shiftChange(df["EShift"])

        if iterShift is not None:
            value = df["Iter"].iloc[iterShift]
            shiftLine.set_xdata([value, value])
            shiftLine.set_visible(True)
        else:
            shiftLine.set_visible(False)

        ax.relim()
        ax.autoscale_view()

    showLive(args, fig, update)

def plotDeterministicNW(args):
    """
    Plot deterministic coefficient populations against iteration.
    """
    df = readDeterministicQMC(args.path)

    iterEs, _ = shiftChange(df["es"])
    iterEsS, _ = shiftChange(df["ess"])

    setStyle()
    plt.figure()

    plt.plot(
        df["iter"],
        df["nwc"],
        label = r"$\|C(\tau)\|_1$",
        linewidth = LINEWIDTH,
    )
    plt.plot(
        df["iter"],
        df["nwsc"],
        label = r"$\|SC(\tau)\|_1$",
        linewidth = LINEWIDTH,
    )

    if iterEs is not None:
        plt.axvline(
            df["iter"].iloc[iterEs],
            linestyle = "--",
            linewidth = LINEWIDTH,
            color = "tab:blue",
        )

    if iterEsS is not None:
        plt.axvline(
            df["iter"].iloc[iterEsS],
            linestyle = "--",
            linewidth = LINEWIDTH,
            color = "tab:orange",
        )

    formatAxes(
        xlabel = r"Iteration / $\tau$",
        ylabel = "Population",
        legend = True,
    )
    finish(args)

def plotDeterministicProjectedShift(args):
    """
    Plot deterministic energy and shift trajectories against iteration.
    """
    if not args.live:
        df = readDeterministicQMC(args.path)

        iterEs, _ = shiftChange(df["es"])
        iterEsS, _ = shiftChange(df["ess"])

        setStyle()
        plt.figure()

        plt.plot(
            df["iter"],
            df["es"],
            label = r"$E_s(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:blue",
        )
        plt.plot(
            df["iter"],
            df["ess"],
            label = r"$E_s^S(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:orange",
        )

        if iterEs is not None:
            plt.axvline(
                df["iter"].iloc[iterEs],
                linestyle = "--",
                linewidth = LINEWIDTH,
                color = "tab:blue",
            )

        if iterEsS is not None:
            plt.axvline(
                df["iter"].iloc[iterEsS],
                linestyle = "--",
                linewidth = LINEWIDTH,
                color = "tab:orange",
            )

        plt.plot(
            df["iter"],
            df["energy"],
            label = r"$E(\tau)$",
            linewidth = LINEWIDTH,
            color = "tab:green",
        )

        formatAxes(
            xlabel = r"Iteration / $\tau$",
            ylabel = "Energy / Ha",
            legend = True,
            legendLoc = "lower right",
        )
        finish(args)
        return

    setStyle()
    fig, ax = plt.subplots()

    lineEs, = ax.plot(
        [],
        [],
        label = r"$E_s(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:blue",
    )
    lineEsS, = ax.plot(
        [],
        [],
        label = r"$E_s^S(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:orange",
    )
    lineEnergy, = ax.plot(
        [],
        [],
        label = r"$E(\tau)$",
        linewidth = LINEWIDTH,
        color = "tab:green",
    )

    shiftEsLine = ax.axvline(
        0.0,
        linestyle = "--",
        linewidth = LINEWIDTH,
        color = "tab:blue",
        visible = False,
    )
    shiftEsSLine = ax.axvline(
        0.0,
        linestyle = "--",
        linewidth = LINEWIDTH,
        color = "tab:orange",
        visible = False,
    )

    formatAxes(
        xlabel = r"Iteration / $\tau$",
        ylabel = "Energy / Ha",
        legend = True,
        legendLoc = "lower right",
    )

    def update():
        df = readDeterministicQMC(args.path).dropna(
            subset = ["iter", "energy", "es", "ess"]
        )
        if df.empty:
            return

        x = df["iter"].to_numpy()

        lineEs.set_data(x, df["es"].to_numpy())
        lineEsS.set_data(x, df["ess"].to_numpy())
        lineEnergy.set_data(x, df["energy"].to_numpy())

        iterEs, _ = shiftChange(df["es"])
        iterEsS, _ = shiftChange(df["ess"])

        if iterEs is not None:
            value = df["iter"].iloc[iterEs]
            shiftEsLine.set_xdata([value, value])
            shiftEsLine.set_visible(True)
        else:
            shiftEsLine.set_visible(False)

        if iterEsS is not None:
            value = df["iter"].iloc[iterEsS]
            shiftEsSLine.set_xdata([value, value])
            shiftEsSLine.set_visible(True)
        else:
            shiftEsSLine.set_visible(False)

        ax.relim()
        ax.autoscale_view()

    showLive(args, fig, update)

def plotShoulder(args):
    """
    Plot the persistent total-to-reference population ratio against population.
    """
    if not args.live:
        df = readQMC(args.path).dropna(
            subset = [
                "NW",
                "NRef",
            ]
        )

        df = df[df["NRef"] != 0.0]

        ratio = df["NW"] / df["NRef"]

        setStyle()
        plt.figure()

        plt.plot(
            df["NW"],
            ratio,
            linewidth = LINEWIDTH,
            color = "tab:blue",
            label = (
                r"$\frac{N_w(\tau)}"
                r"{N_{w,\mathrm{ref}}(\tau)}$"
            ),
        )

        formatAxes(
            xlabel = r"$N_w(\tau)$",
            ylabel = (
                r"$N_w(\tau)"
                r"/N_{w,\mathrm{ref}}(\tau)$"
            ),
            legend = True,
        )

        finish(args)
        return

    setStyle()
    fig, ax = plt.subplots()

    lineRatio, = ax.plot(
        [],
        [],
        linewidth = LINEWIDTH,
        color = "tab:blue",
        label = (
            r"$\frac{N_w(\tau)}"
            r"{N_{w,\mathrm{ref}}(\tau)}$"
        ),
    )

    formatAxes(
        xlabel = r"$N_w(\tau)$",
        ylabel = (
            r"$N_w(\tau)"
            r"/N_{w,\mathrm{ref}}(\tau)$"
        ),
        legend = True,
    )

    def update():
        df = readQMC(args.path).dropna(
            subset = [
                "NW",
                "NRef",
            ]
        )

        df = df[df["NRef"] != 0.0]

        if df.empty:
            return

        ratio = df["NW"] / df["NRef"]

        lineRatio.set_data(
            df["NW"].to_numpy(),
            ratio.to_numpy(),
        )

        ax.relim()
        ax.autoscale_view()

    showLive(args, fig, update)

def plotReferenceOverlap(args):
    """
    Plot the normalised projected-energy denominator against iteration.
    """
    if not args.live:
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EProjDen",
                "NW",
            ]
        )

        df = df[df["NW"] != 0.0]

        overlap = df["EProjDen"] / df["NW"]
        average = overlap.rolling(
            window = args.window,
            min_periods = 1,
        ).mean()

        setStyle()
        plt.figure()

        plt.plot(
            df["Iter"],
            overlap,
            linewidth = 1,
            alpha = 0.25,
            color = "tab:blue",
            label = (
                r"$\frac{E_{\mathrm{ProjDen}}(\tau)}{N_w(\tau)}$"
            ),
        )

        plt.plot(
            df["Iter"],
            average,
            linewidth = LINEWIDTH,
            color = "tab:orange",
            label = (
                rf"Rolling mean, ${args.window}$ samples"
            ),
        )

        formatAxes(
            xlabel = r"Iteration / $\tau$",
            ylabel = (
                r"$\frac{\langle \Psi_{\mathrm{Ref}} | \Psi(\tau) \rangle}"
                r"{N_w(\tau)}$"
            ),
            legend = True,
        )

        plt.grid(True)
        finish(args)
        return

    setStyle()
    fig, ax = plt.subplots()

    lineOverlap, = ax.plot(
        [],
        [],
        linewidth = 1,
        alpha = 0.7,
        color = "tab:blue",
        label = (
            r"$\frac{\langle \Psi_{\mathrm{Ref}} | \Psi(\tau) \rangle}"
            r"{N_w(\tau)}$"
        ),
    )

    lineAverage, = ax.plot(
        [],
        [],
        linewidth = LINEWIDTH,
        color = "tab:orange",
        label = (
            r"$\overline{\frac{\langle \Psi_{\mathrm{Ref}}"
            r" | \Psi(\tau) \rangle}{N_w(\tau)}}$"
        ),
    )

    formatAxes(
        xlabel = r"Iteration / $\tau$",
        ylabel = (
            r"$\frac{\langle \Psi_{\mathrm{Ref}} | \Psi(\tau) \rangle}"
            r"{N_w(\tau)}$"
        ),
        legend = True,
    )

    ax.grid(True)

    def update():
        df = readQMC(args.path).dropna(
            subset = [
                "Iter",
                "EProjDen",
                "NW",
            ]
        )

        df = df[df["NW"] != 0.0]

        if df.empty:
            return

        x = df["Iter"].to_numpy()
        overlap = df["EProjDen"] / df["NW"]
        average = overlap.rolling(
            window = args.window,
            min_periods = 1,
        ).mean()

        lineOverlap.set_data(
            x,
            overlap.to_numpy(),
        )

        lineAverage.set_data(
            x,
            average.to_numpy(),
        )

        ax.relim()
        ax.autoscale_view()

    showLive(args, fig, update)

def setStyle():
    """
    Choose a uniform font for all plotting purposes.
    """
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.figsize"] = (22, 10)

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

    p = subparsers.add_parser("deterministic-population")
    p.add_argument("path", type = Path)
    addCommonArgs(p)
    p.set_defaults(func = plotDeterministicNW)
    
    p = subparsers.add_parser("deterministic-projected-shift")
    p.add_argument("path", type = Path)
    addTrajectoryArgs(p)
    p.set_defaults(func = plotDeterministicProjectedShift)

    p = subparsers.add_parser("energy")
    p.add_argument("path", type = Path)
    p.add_argument("tables_pos", nargs = "*", type = Path)
    p.add_argument("--table", dest = "tables_opt", action = "append", type = Path, default = [])
    p.add_argument("--table-label", action = "append", default = [])
    p.add_argument(
        "--inset",
        nargs = 4,
        type = float,
        metavar = ("XMIN", "XMAX", "YMIN", "YMAX"),
        default = None,
        help = "Add an inset with zoom limits XMIN XMAX YMIN YMAX.",
    )
    p.add_argument(
        "--inset-side",
        choices = ["right", "left", "above", "below"],
        default = "right",
        help = "Side of the zoom region where the inset is placed.",
    )
    p.add_argument(
        "--inset-location",
        nargs = 2,
        type = float,
        metavar = ("LEFT", "BOTTOM"),
        default = [0.72, 0.03],
        help = "Inset lower-left position in axes-fraction coordinates.",
    )
    p.add_argument(
        "--inset-size",
        nargs = 2,
        type = float,
        metavar = ("WIDTH", "HEIGHT"),
        default = [0.25, 0.25],
        help = "Inset size in axes-fraction coordinates.",
    )
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
    addTrajectoryArgs(p)
    p.set_defaults(func = plotNW)
    
    p = subparsers.add_parser("projected-shift")
    p.add_argument("path", type = Path)
    addTrajectoryArgs(p)
    p.set_defaults(func = plotProjectedShift)
    
    p = subparsers.add_parser("shoulder")
    p.add_argument("path", type = Path)
    addTrajectoryArgs(p)
    p.set_defaults(func = plotShoulder)

    p = subparsers.add_parser("reference-overlap")
    p.add_argument("path", type = Path)
    p.add_argument(
        "--window",
        type = int,
        default = 1000,
        help = "Number of output samples used in the rolling mean.",
    )
    addTrajectoryArgs(p)
    p.set_defaults(func = plotReferenceOverlap)

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
