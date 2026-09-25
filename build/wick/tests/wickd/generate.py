# generate.py

from pathlib import Path
import argparse
import re

import wickd as w

# Spin-orbital cluster excitation types as "cre ... ann ..." strings.
CLUSTERTYPES = [
    "a+ o",
    "v+ a",
    "a+ a",
    "v+ o",
    "a+ v+ a o",
    "v+ v+ a o",
    "a+ v+ o o",
    "a+ a+ o o",
    "a+ a+ a o",
    "a+ v+ a a",
    "v+ v+ a a",
    "a+ a+ a a",
]

# Residual projectors tau^dagger for every spin-orbital excitation class.
PROJECTORS = {
    "CToA": "o+ a",
    "AToV": "a+ v",
    "AToA": "a+ a",
    "CToV": "o+ v",
    "CAToAV": "o+ a+ a v",
    "CAToVV": "o+ a+ v v",
    "CCToAV": "o+ o+ a v",
    "CCToAA": "o+ o+ a a",
    "CAToAA": "o+ a+ a a",
    "AAToAV": "a+ a+ a v",
    "AAToVV": "a+ a+ v v",
    "AAToAA": "a+ a+ a a",
}

# Largest cumulant rank kept in contractions.
MAXCUMULANT = 4

# All required regex goes here.
TENSORREGEX = re.compile(r"([A-Za-z]+\d*)\^\{([^}]*)\}_\{([^}]*)\}")
CONTRACTIONREGEX = re.compile(r"^(gamma1|eta1|lambda\d+)$")


def setupSpaces():
    """
    Define core (occupied), active (general) and virtual (unoccupied) spaces.
    """
    w.reset_space()
    w.add_space("o", "fermion", "occupied", [f"i{n}" for n in range(24)])
    w.add_space("a", "fermion", "general", [f"u{n}" for n in range(24)])
    w.add_space("v", "fermion", "unoccupied", [f"e{n}" for n in range(24)])


def isKept(line: str) -> bool:
    """
    Check that a term is connected and has no contraction inside {T T}.
    """
    tensors = [
        (label, set(x for x in (upper + "," + lower).split(",") if x))
        for label, upper, lower in TENSORREGEX.findall(line)
    ]
    ops = [t for t in tensors if not CONTRACTIONREGEX.match(t[0])]
    contractions = [t for t in tensors if CONTRACTIONREGEX.match(t[0])]
    nodes = ops + contractions
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    # Join tensors that share an index.
    owner = {}
    for n, (_, indices) in enumerate(nodes):
        for i in indices:
            if i in owner:
                parent[find(n)] = find(owner[i])
            else:
                owner[i] = n

    if len({find(n) for n in range(len(ops))}) != 1:
        return False

    # Both amplitudes form one normal-ordered string, so nothing contracts only them.
    amplitudes = [indices for label, indices in ops if label == "t"]
    if len(amplitudes) == 2:
        both = amplitudes[0] | amplitudes[1]
        if amplitudes[0] & amplitudes[1]:
            return False
        if any(indices <= both for _, indices in contractions):
            return False
    return True


def residualTerms(excitation: str, order: int) -> list:
    """
    Return the kept Wick&D residual terms of one class at one order in T.
    """
    setupSpaces()
    bra = w.op("R", [PROJECTORS[excitation]])
    hamiltonian = w.utils.gen_op("f", 1, "oav", "oav") + w.utils.gen_op(
        "v", 2, "oav", "oav"
    )
    cluster = w.op("t", CLUSTERTYPES)

    product = bra @ hamiltonian
    for _ in range(order):
        product = product @ cluster
    if order == 2:
        product = w.rational(1, 2) * product

    theorem = w.WickTheorem()
    theorem.set_max_cumulant(MAXCUMULANT)
    theorem.set_single_threaded(True)
    expr = theorem.contract(product, 0, 0)

    return [line for line in str(expr).splitlines() if line.strip() and isKept(line)]


def main():
    """
    Write one reference file per requested class and order.
    """
    parser = argparse.ArgumentParser(description="Generate Wick&D GNOCC residuals.")
    parser.add_argument("order", type=int, help="Order in T (0, 1 or 2).")
    parser.add_argument("classes", nargs="*", help="Excitation classes (default all).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    for excitation in args.classes or list(PROJECTORS):
        lines = residualTerms(excitation, args.order)
        path = here / f"r{args.order}_{excitation}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"r{args.order} {excitation}: {len(lines)} terms", flush=True)


if __name__ == "__main__":
    main()
