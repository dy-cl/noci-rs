# benchmarks.py

"""Run the short H6 benchmarks and publish SVG badges and a per-CPU table to GitHub Pages."""

import argparse
import datetime as dt
import html
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path

BENCHMARKS = {
    "sapply": (
        "H6 cc-pVDZ · NOCISDT(3) · SApply",
        "inputs/benchmarks/H6_cc-pVDZ_NOCISDT3_1_5_SApply.lua",
    ),
    "bapply": (
        "H6 cc-pVDZ · NOCISDT(3) · BApply",
        "inputs/benchmarks/H6_cc-pVDZ_NOCISDT3_1_5_BApply.lua",
    ),
    "pt2": (
        "H6 cc-pVDZ · NOCISD(3) · PT2",
        "inputs/benchmarks/H6_cc-pVDZ_NOCISD3_1_5_PT2.lua",
    ),
}


def git(args, cwd, check=True):
    """Run Git in the requested directory and capture its output."""
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=check
    )


def memoryGiB():
    """Read the runner's effective memory limit in GiB."""
    total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
                break
    except OSError:
        pass

    # A hosted runner may expose the host's MemTotal through /proc.
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            limit = int(Path(path).read_text().strip())
            if limit < 1 << 60:
                total = min(total, limit) if total else limit
        except (OSError, ValueError):
            pass

    return round(total / (1024**3), 1) if total else None


def systemSpecs():
    """Collect the runner OS, CPU model, available CPUs, and memory."""
    cpu = platform.processor() or "Unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    osName = platform.system()
    try:
        fields = {}
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value.strip('"')
        osName = fields.get("PRETTY_NAME", osName)
    except OSError:
        pass

    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )

    return {"os": osName, "cpu": cpu, "vcpus": cores, "memory_gib": memoryGiB()}


def runBenchmarks(binary, resultPath, logDir, timeout):
    """Measure each H6 input and save results and separate output logs."""
    logDir.mkdir(parents=True, exist_ok=True)
    result = {
        "commit": os.environ.get("GITHUB_SHA"),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "system": systemSpecs(),
        "settings": {
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "automatic"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "benchmarks": {},
    }

    for key, (label, inputPath) in BENCHMARKS.items():
        status = "ok"
        message = None
        logPath = logDir / f"{key}.log"

        # PT2's disk-backed matrix is several GiB; remove it after measuring.
        scratchDir = (
            tempfile.TemporaryDirectory(prefix="noci-pt2-")
            if key == "pt2"
            else nullcontext(None)
        )
        with scratchDir as scratch:
            start = time.perf_counter()
            try:
                with logPath.open("w") as log:
                    completed = subprocess.run(
                        [str(binary), str(Path(inputPath).resolve())],
                        cwd=scratch,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                    )
                if completed.returncode != 0:
                    status = "failed"
                    message = f"Exit status {completed.returncode}"
            except (OSError, subprocess.TimeoutExpired) as error:
                status = "unavailable"
                message = str(error)
                with logPath.open("a") as log:
                    log.write(f"\n{message}\n")
            elapsed = time.perf_counter() - start

        result["benchmarks"][key] = {
            "label": label,
            "input": inputPath,
            "status": status,
            "time_seconds": round(elapsed, 3) if status == "ok" else None,
            "error": message,
        }

        print(f"{label}: {elapsed:.2f} s ({status})", flush=True)

    resultPath.parent.mkdir(parents=True, exist_ok=True)
    resultPath.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def readPrevious(path):
    """Read an earlier run, or return an empty result when unavailable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def machine(run):
    """Identify the machine type of a run by its CPU model and available CPUs."""
    system = run.get("system", {})
    return (system.get("cpu") or "Unknown CPU", system.get("vcpus"))


def readHistory():
    """Read the runs published on GitHub Pages, or none before the first deployment."""
    owner, name = os.environ["GITHUB_REPOSITORY"].split("/")
    url = f"https://{owner}.github.io/{name}/runs.json"

    # A custom domain behind Cloudflare refuses Python's default user agent.
    request = urllib.request.Request(url, headers={"User-Agent": "noci-rs-benchmarks"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    return []


def previousOnMachine(runs, current):
    """Find the latest run before `current` on the same machine type."""
    earlier = [
        run
        for run in runs
        if machine(run) == machine(current)
        and run.get("commit") != current.get("commit")
        and (run.get("recorded_at") or "") <= (current.get("recorded_at") or "")
    ]
    return earlier[-1] if earlier else {}


def compareTimes(current, previous):
    """Compare a time with the previous one on the same machine type, with its colour."""
    currentTime = current.get("time_seconds") if current.get("status") == "ok" else None
    previousTime = (
        previous.get("time_seconds") if previous.get("status") == "ok" else None
    )

    if currentTime is None:
        return "unavailable", None, "#6e7781"
    if previousTime is None:
        return f"{currentTime:.2f} s", None, "#6e7781"
    if round(currentTime, 2) < round(previousTime, 2):
        return f"↑ {currentTime:.2f} s", f"{previousTime:.2f} s", "#1a7f37"
    if round(currentTime, 2) > round(previousTime, 2):
        return f"↓ {currentTime:.2f} s", f"{previousTime:.2f} s", "#cf222e"
    return f"→ {currentTime:.2f} s", f"{previousTime:.2f} s", "#6e7781"


def formatSpecs(specs):
    """Format the runner specifications for their shared badge."""
    parts = [specs.get("os") or "Unknown OS", specs.get("cpu") or "Unknown CPU"]
    if specs.get("vcpus"):
        parts.append(f"{specs['vcpus']} vCPU")
    if specs.get("memory_gib"):
        parts.append(f"{specs['memory_gib']:g} GiB RAM")

    return " · ".join(parts)


def renderPill(label, message, color, labelWidth, width, details):
    """Render a compact two-part badge with an accessible description."""
    label = html.escape(label)
    message = html.escape(message)
    details = html.escape(details)
    messageWidth = width - labelWidth

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="24" viewBox="0 0 {width} 24" role="img" aria-label="{details}">
  <title>{details}</title>
  <rect width="{width}" height="24" rx="4" fill="#555"/>
  <rect x="{labelWidth}" width="{messageWidth}" height="24" rx="4" fill="{color}"/>
  <rect x="{labelWidth}" width="4" height="24" fill="{color}"/>
  <text x="12" y="16" fill="#fff" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">{label}</text>
  <text x="{labelWidth + 12}" y="16" fill="#fff" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">{message}</text>
</svg>
"""


def renderBadge(label, current, previous, specs):
    """Render a time badge compared with the previous run on the same machine type."""
    time, previousTime, color = compareTimes(current, previous)
    comparison = f"{time} (previous {previousTime})" if previousTime else time

    details = f"{label}: {comparison}; {formatSpecs(specs)}"
    return renderPill(label, comparison, color, 260, 530, details)


def renderRunnerBadge(specs):
    """Render the common runner specifications once below the timings."""
    details = formatSpecs(specs)
    return renderPill("runner", details, "#57606a", 76, 820, f"Runner: {details}")


def renderMachineTable(runs):
    """Render one row per machine type with its latest times against its previous run."""
    names = [label.split(" · ")[-1] for label, _ in BENCHMARKS.values()]

    # Machine types with the most recently measured first.
    latest = {}
    for run in runs:
        latest[machine(run)] = run
    rows = []
    for run in sorted(
        latest.values(), key=lambda run: run.get("recorded_at") or "", reverse=True
    ):
        previous = previousOnMachine(runs, run)
        cells = []
        for key in BENCHMARKS:
            time, previousTime, color = compareTimes(
                run.get("benchmarks", {}).get(key, {}),
                previous.get("benchmarks", {}).get(key, {}),
            )
            cells.append((f"{time} ({previousTime})" if previousTime else time, color))
        cpu, vcpus = machine(run)
        rows.append((f"{cpu} · {vcpus} vCPU" if vcpus else cpu, cells))

    # Column widths from an average glyph width of the 12 px badge font.
    glyph = 7
    labelWidth = glyph * max(len("runner CPU"), *(len(label) for label, _ in rows)) + 24
    widths = [
        glyph * max(len(name), *(len(cells[k][0]) for _, cells in rows)) + 24
        for k, name in enumerate(names)
    ]
    width = labelWidth + sum(widths)
    height = 28 * len(rows) + 24

    def row(y, label, cells):
        """Render one row as a pill with a label and one coloured cell per benchmark."""
        parts = [
            f'  <rect y="{y}" width="{width}" height="24" rx="4" fill="#555"/>',
            f'  <text x="12" y="{y + 16}" fill="#fff" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">{html.escape(label)}</text>',
        ]
        x = labelWidth
        for k, (message, color) in enumerate(cells):
            # The last cell rounds the row's right end, as the message of a badge does.
            last = k == len(cells) - 1
            corner = ' rx="4"' if last else ""
            parts.append(
                f'  <rect x="{x + 1}" y="{y}" width="{widths[k] - 1}" height="24"{corner} fill="{color}"/>'
            )
            if last:
                parts.append(
                    f'  <rect x="{x + 1}" y="{y}" width="4" height="24" fill="{color}"/>'
                )
            parts.append(
                f'  <text x="{x + 12}" y="{y + 16}" fill="#fff" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">{html.escape(message)}</text>'
            )
            x += widths[k]
        return parts

    details = "Benchmark times by runner CPU"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{details}">',
        f"  <title>{details}</title>",
        *row(0, "runner CPU", [(name, "#57606a") for name in names]),
    ]
    for r, (label, cells) in enumerate(rows):
        lines.extend(row(28 * (r + 1), label, cells))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def publish(resultPath, repo, siteDir):
    """Add this commit's run to the history and build the Pages site when it is the main tip."""
    sha = os.environ["GITHUB_SHA"]
    current = readPrevious(resultPath)
    if not current:
        current = {"commit": sha, "system": systemSpecs(), "benchmarks": {}}
    current["commit"] = sha

    # One run per commit, a rerun replacing the earlier one, in recording order.
    runs = [run for run in readHistory() if run.get("commit") != sha] + [current]
    runs.sort(key=lambda run: run.get("recorded_at") or "")

    # A superseded workflow must not replace newer badges.
    mainTip = git(["ls-remote", "origin", "refs/heads/main"], repo).stdout.split()[0]
    deploy = sha == mainTip
    if deploy:
        badgeDir = siteDir / "badges"
        badgeDir.mkdir(parents=True, exist_ok=True)
        previous = previousOnMachine(runs, current)
        for key, (label, _) in BENCHMARKS.items():
            currentRun = current.get("benchmarks", {}).get(key, {})
            previousRun = previous.get("benchmarks", {}).get(key, {})
            badge = renderBadge(label, currentRun, previousRun, current["system"])
            (badgeDir / f"{key}.svg").write_text(badge)

        (badgeDir / "runner.svg").write_text(renderRunnerBadge(current["system"]))
        (badgeDir / "machines.svg").write_text(renderMachineTable(runs))
        serialized = json.dumps(current, indent=2, ensure_ascii=False) + "\n"
        (siteDir / "latest.json").write_text(serialized)
        history = json.dumps(runs, indent=2, ensure_ascii=False) + "\n"
        (siteDir / "runs.json").write_text(history)

    # Tell the workflow whether to deploy the site.
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"deploy={str(deploy).lower()}\n")


def main():
    """Parse the run or publish command and perform the requested action."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--result", type=Path, required=True)
    run.add_argument("--logs", type=Path, required=True)
    run.add_argument("--timeout", type=int, default=120)
    post = subcommands.add_parser("publish")
    post.add_argument("--result", type=Path, required=True)
    post.add_argument("--repo", type=Path, default=Path.cwd())
    post.add_argument("--site", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        runBenchmarks(args.binary.resolve(), args.result, args.logs, args.timeout)
    else:
        publish(args.result, args.repo.resolve(), args.site.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Benchmark report failed: {error}", file=sys.stderr)
        sys.exit(1)
