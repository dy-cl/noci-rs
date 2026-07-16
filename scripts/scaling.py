#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import os
import subprocess
import sys
import time

def parseList(value: str):
    """
    Parse a comma-separated list of positive integers.
    """
    values = [int(x) for x in value.split(",")]

    if any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("Values must be positive integers")

    return values

def runCase(args, mpiRanks: int, rayonThreads: int):
    """
    Run one MPI/Rayon benchmark and return the wall time in milliseconds.
    """
    nodes = math.ceil(mpiRanks / args.tasksPerNode)
    repoRoot = args.bin.resolve().parents[2]

    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(rayonThreads)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"

    if "SLURM_JOB_ID" in environment:
        command = [
            "srun",
            "--mpi=pmix",
            "--nodes", str(nodes),
            "--ntasks", str(mpiRanks),
            "--ntasks-per-node", str(args.tasksPerNode),
            "--cpus-per-task", str(rayonThreads),
            "--distribution=block:block",
            "--cpu-bind=cores",
            "--exclusive",
            str(args.bin.resolve()),
            str(args.input.resolve()),
        ]
    else:
        command = [
            "mpirun",
            "-np", str(mpiRanks),
            "bash",
            "-lc",
            (
                "localRank=${OMPI_COMM_WORLD_LOCAL_RANK:-"
                "${MPI_LOCALRANKID:-${SLURM_LOCALID:-0}}}; "
                f"start=$((localRank * {rayonThreads})); "
                f"end=$((start + {rayonThreads} - 1)); "
                "cpus=$(seq -s, $start $end); "
                f'exec taskset --cpu-list "$cpus" '
                f'env RAYON_NUM_THREADS={rayonThreads} '
                "OPENBLAS_NUM_THREADS=1 "
                "OMP_NUM_THREADS=1 "
                f'"{args.bin.resolve()}" "{args.input.resolve()}"'
            ),
        ]

    print(
        f"Running MPI={mpiRanks}, "
        f"Rayon/rank={rayonThreads}, "
        f"nodes={nodes}",
        file = sys.stderr,
        flush = True,
    )

    start = time.perf_counter()

    subprocess.run(
        command,
        cwd = repoRoot,
        env = environment,
        stdout = subprocess.DEVNULL,
        check = True,
    )

    return round(1000.0 * (time.perf_counter() - start))

def buildParser():
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type = Path, required = True)
    parser.add_argument("--input", type = Path, required = True)
    parser.add_argument("--rayon", type = parseList, required = True)
    parser.add_argument("--mpi", type = parseList, required = True)
    parser.add_argument(
        "--tasks-per-node",
        dest = "tasksPerNode",
        type = int,
        required = True,
    )
    parser.add_argument(
        "--cpus-per-node",
        dest = "cpusPerNode",
        type = int,
        required = True,
    )
    parser.add_argument("--out", type = Path)

    return parser

def main():
    """
    Run all valid MPI/Rayon combinations.
    """
    args = buildParser().parse_args()

    if not args.bin.is_file() or not os.access(args.bin, os.X_OK):
        raise FileNotFoundError(f"Executable not found: {args.bin}")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.out is not None:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        output = open(args.out, "w")
    else:
        output = sys.stdout

    try:
        print(
            "# RAYON CPUs\t# MPI RANKS\tWall Time (ms)",
            file = output,
            flush = True,
        )

        for mpiRanks in args.mpi:
            for rayonThreads in args.rayon:
                ranksPerNode = min(mpiRanks, args.tasksPerNode)

                if ranksPerNode * rayonThreads > args.cpusPerNode:
                    continue

                wallTime = runCase(args, mpiRanks, rayonThreads)

                print(
                    f"{rayonThreads}\t{mpiRanks}\t{wallTime}",
                    file = output,
                    flush = True,
                )
    finally:
        if output is not sys.stdout:
            output.close()

if __name__ == "__main__":
    main()
