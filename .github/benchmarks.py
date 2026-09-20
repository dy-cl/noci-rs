# benchmarks.py

"""Run the short H6 benchmarks and publish informational SVG badges."""

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
RESULTS_BRANCH = "benchmark-results"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


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
    """Render a time badge with the preceding commit's comparison."""
    currentTime = current.get("time_seconds") if current.get("status") == "ok" else None
    previousTime = (
        previous.get("time_seconds") if previous.get("status") == "ok" else None
    )

    if currentTime is None:
        comparison, color = "unavailable", "#6e7781"
    elif previousTime is None:
        comparison, color = f"{currentTime:.2f} s", "#6e7781"
    elif round(currentTime, 2) < round(previousTime, 2):
        comparison, color = (
            f"↑ {currentTime:.2f} s (previous {previousTime:.2f} s)",
            "#1a7f37",
        )
    elif round(currentTime, 2) > round(previousTime, 2):
        comparison, color = (
            f"↓ {currentTime:.2f} s (previous {previousTime:.2f} s)",
            "#cf222e",
        )
    else:
        comparison, color = (
            f"→ {currentTime:.2f} s (previous {previousTime:.2f} s)",
            "#6e7781",
        )

    details = f"{label}: {comparison}; {formatSpecs(specs)}"
    return renderPill(label, comparison, color, 260, 530, details)


def renderRunnerBadge(specs):
    """Render the common runner specifications once below the timings."""
    details = formatSpecs(specs)
    return renderPill("runner", details, "#57606a", 76, 820, f"Runner: {details}")


def publish(resultPath, repo, branchDir):
    """Store this commit's run and refresh badges when it is the main tip."""
    sha = os.environ["GITHUB_SHA"]
    previousCommit = git(["rev-parse", "HEAD^"], repo, check=False)
    previousSha = (
        previousCommit.stdout.strip() if previousCommit.returncode == 0 else None
    )

    # Fetch the report branch, or create an orphan branch for its first run.
    query = ["ls-remote", "--exit-code", "--heads", "origin", RESULTS_BRANCH]
    remoteBranch = git(query, repo, check=False)
    if remoteBranch.returncode == 0:
        sourceRef = f"refs/heads/{RESULTS_BRANCH}"
        remoteRef = f"refs/remotes/origin/{RESULTS_BRANCH}"
        git(["fetch", "origin", f"{sourceRef}:{remoteRef}"], repo)
        git(["worktree", "add", "--detach", str(branchDir), remoteRef], repo)
    elif remoteBranch.returncode == 2:
        git(["worktree", "add", "--detach", str(branchDir), "HEAD"], repo)
        git(["switch", "--orphan", RESULTS_BRANCH], branchDir)
    else:
        raise RuntimeError(
            remoteBranch.stderr.strip() or "Could not query results branch"
        )

    # Keep every commit's JSON so an exact parent result can be compared.
    current = readPrevious(resultPath)
    if not current:
        current = {"commit": sha, "system": systemSpecs(), "benchmarks": {}}
    current["commit"] = sha

    previous = (
        readPrevious(branchDir / "runs" / f"{previousSha}.json") if previousSha else {}
    )

    runDir = branchDir / "runs"
    runDir.mkdir(exist_ok=True)
    serialized = json.dumps(current, indent=2, ensure_ascii=False) + "\n"
    (runDir / f"{sha}.json").write_text(serialized)

    # A superseded workflow may save its run, but must not replace newer badges.
    mainTip = git(["ls-remote", "origin", "refs/heads/main"], repo).stdout.split()[0]
    if sha == mainTip:
        badgeDir = branchDir / "badges"
        badgeDir.mkdir(exist_ok=True)
        for key, (label, _) in BENCHMARKS.items():
            currentRun = current.get("benchmarks", {}).get(key, {})
            previousRun = previous.get("benchmarks", {}).get(key, {})
            badge = renderBadge(label, currentRun, previousRun, current["system"])
            (badgeDir / f"{key}.svg").write_text(badge)

        (badgeDir / "runner.svg").write_text(renderRunnerBadge(current["system"]))
        (branchDir / "latest.json").write_text(serialized)

    # Publish the informational results independently of the source branch.
    git(["config", "user.name", "github-actions[bot]"], branchDir)
    git(["config", "user.email", BOT_EMAIL], branchDir)
    git(["add", "."], branchDir)
    changes = git(["diff", "--cached", "--quiet"], branchDir, check=False)
    if changes.returncode == 1:
        git(["commit", "-m", f"Record H6 benchmarks for {sha[:12]}"], branchDir)
        git(["push", "origin", f"HEAD:refs/heads/{RESULTS_BRANCH}"], branchDir)


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
    post.add_argument("--worktree", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        runBenchmarks(args.binary.resolve(), args.result, args.logs, args.timeout)
    else:
        publish(args.result, args.repo.resolve(), args.worktree.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Benchmark report failed: {error}", file=sys.stderr)
        sys.exit(1)
