from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product as cartesianProduct
from pathlib import Path

from core import (
    Expr,
    Idx,
    Op,
    Ref,
    SPINS,
    Space,
    Tensor,
    Term,
    add,
    combine,
    partitions,
    partitionSign,
    prod,
    scale,
    zero,
)

@dataclass(frozen = True)
class CumulantRank:
    """
    One generated spin-free cumulant. Stores rank and symbolic index names 
    used to generate the Rust cumulants builder.

    Notation:
        \Lambda_k = \Gamma_k - D_k (disconnected).

    Examples:
        CumulantRank(3, ("p", "q", "r"), ("s", "t", "u")) represents generation of cumulants3.rs.
    """
    rank: int
    upper: tuple[str, ...]
    lower: tuple[str, ...]

RANKS = {
    1: CumulantRank(1, ("p",), ("q",)),
    2: CumulantRank(2, ("p", "q"), ("r", "s")),
    3: CumulantRank(3, ("p", "q", "r"), ("s", "t", "u")),
    4: CumulantRank(4, ("p", "q", "r", "w"), ("s", "t", "u", "v")),
}

def activeIdx(name: str) -> Idx:
    """
    Build one active orbital index.

    Notation:
        p \in in A

    Examples:
        activeIdx("p") returns the active index p.
    """
    return Idx(name, Space.ACTIVE)

def gammaSpinString(rank: CumulantRank, spins: tuple[str, ...]) -> tuple[Op, ...]:
    """
    Build one spin-orbital component of a spin-free k-RDM.

    Notation:
        Gamma^{p q ...}_{r s ...} 
        = sum_{\sigma \tau \cdots} \langle \Phi | a^dagger_{p \sigma} a^dagger_{q \tau} \cdots a_{s tau} a_{r sigma} | \Phi \rangle

    Examples:
        For rank 2, this returns a^dagger_{p \sigma} a^dagger_{q \tau} a_{s \tau} a_{r \sigma}.
    """
    upper = tuple(activeIdx(name) for name in rank.upper)
    lower = tuple(activeIdx(name) for name in rank.lower)

    creators = tuple(
        Op("create", idx, spin, 0)
        for idx, spin in zip(upper, spins)
    )

    annihilators = tuple(
        Op("annihilate", idx, spin, 0)
        for idx, spin in zip(reversed(lower), reversed(spins))
    )

    return creators + annihilators

def gammaDisconnectedExpr(rank: CumulantRank) -> Expr:
    """
    Build the disconnected part of a spin-free k-RDM.

    Notation:
        \Gamma_k = \Lambda_k + D_k (disconnected)
        D_k = \sum_{P, P > 1} \sign(P) \prod_{B \in P} \kappa(B).

    Examples:
        For rank 3, this generates the expression subtracted in cumulants3.rs.
    """
    # One-body cumulant is just the one-body RDM thus no disconnected part.
    if rank.rank == 1:
        return zero()

    ref = Ref()
    out = zero()
    
    # Sum over alpha and beta spins. 
    for spins in cartesianProduct(SPINS, repeat = rank.rank):
        # Construct spin-orbital string a_{p \sigma}^\dagger, a_{q \tau}^\dagger \cdots a_{s \tau} a_{r \sigma}.
        ops = gammaSpinString(rank, spins)
        
        # Get and enumerate all paritions.
        positions = tuple(range(len(ops)))
        for partition in partitions(positions):
            # Skip full connected partition as this is \Lambda_k and we want only D_k.
            if len(partition) == 1:
                continue

            factors = []
            valid = True
            
            # Evaluate all blocks B in \kappa(B) belonging to parition P.
            for block in partition:
                value = ref.kappa(
                    tuple(ops[i] for i in block)
                )
                
                # For a zero block we can skip the whole partition.
                if not value:
                    valid = False
                    break

                factors.append(value)

            if not valid:
                continue
            
            # Perform sum \sum_{P, P > 1} \sign(P) \prod_{B \in P} \kappa(B). 
            out = add(
                out,
                scale(
                    prod(tuple(factors)),
                    partitionSign(partition),
                ),
            )

    return combine(out)

def rustCoeff(coeff) -> str:
    """
    Emit one Rust scalar coefficient.

    Notation:
        <T as From<f64>>::from(a / b)

    Examples:
        Fraction(1, 2) becomes <T as From<f64>>::from(1.0 / 2.0).
    """
    if coeff.denominator == 1:
        return f"<T as From<f64>>::from({coeff.numerator}.0)"

    return f"<T as From<f64>>::from({coeff.numerator}.0 / {coeff.denominator}.0)"

def rustTensor(tensor: Tensor) -> str:
    """
    Emit one inline Rust tensor access for generated cumulant code.

    Notation:
        Gamma1 -> lambda1.get(...)
        Lambda2 -> lambda2.get(...)
        Lambda3 -> lambda3.get(...)

    Examples:
        Tensor("Lambda2", (p, q), (r, s)) becomes lambda2.get(&[p, q], &[r, s]).
    """
    upper = ", ".join(idx.name for idx in tensor.upper)
    lower = ", ".join(idx.name for idx in tensor.lower)

    if tensor.name == "Gamma1":
        return f"lambda1.get(&[{upper}], &[{lower}])"

    if tensor.name == "Lambda2":
        return f"lambda2.get(&[{upper}], &[{lower}])"

    if tensor.name == "Lambda3":
        return f"lambda3.get(&[{upper}], &[{lower}])"

    if tensor.name == "Lambda4":
        raise ValueError("disconnected cumulant expression must not contain Lambda4")

    if tensor.name == "Theta":
        raise ValueError("active cumulant expression should not contain Theta")

    raise ValueError(f"unknown tensor {tensor.name}")

def rustTermBody(term: Term) -> str:
    """
    Emit the body of one Rust product term.

    Notation:
        c A B C

    Examples:
        (1/2) Gamma1 Lambda2 becomes c(1.0 / 2.0) * g1(...) * l2(...).
    """
    coeff = abs(term.coeff)
    factors = []

    if term.deltas:
        raise ValueError("active cumulant expression should not contain deltas")

    if coeff != 1:
        factors.append(rustCoeff(coeff))

    factors.extend(
        rustTensor(tensor)
        for tensor in term.tensors
    )

    if not factors:
        return "c(1.0)"

    return " * ".join(factors)

def rustExpr(expr: Expr, indent: str) -> str:
    """
    Emit one Rust expression from symbolic terms.

    Notation:
        \sum_i c_i t_i

    Examples:
        Gamma1 Gamma1 - (1/2) Gamma1 Gamma1 is emitted as a multiline Rust expression.
    """
    if not expr:
        return f"{indent}c(0.0)"

    lines = []

    for i, term in enumerate(expr):
        sign = "-" if term.coeff < 0 else "+"
        body = rustTermBody(term)
        
        # No leading plus on first line.
        if i == 0:
            lines.append(f"{indent}- {body}" if sign == "-" else f"{indent}{body}")
        else:
            lines.append(f"{indent}{sign} {body}")

    return "\n".join(lines)

def flatIndex(names: tuple[str, ...], gammaName: str) -> str:
    """
    Emit a flat tensor index without pointless outer parentheses.

    Notation:
        ((p * n + q) * n + r) ...

    Examples:
        flatIndex(("p", "q"), "gamma1") gives p * gamma1.n + q.
    """
    expr = f"{names[0]} * {gammaName}.n + {names[1]}"

    for name in names[2:]:
        expr = f"({expr}) * {gammaName}.n + {name}"

    return expr

def rustDataNames(rank: CumulantRank) -> tuple[str, ...]:
    """
    Return names used to index the input RDM.

    Notation:
        ranks 1 and 2 use full-space active-mapped indices.

    Examples:
        rank 2 returns pp, qq, rr, ss.
    """
    names = rank.upper + rank.lower
    
    # Rank 1 and 2 RDMs stored in full NO basis.
    if rank.rank <= 2:
        return tuple(name + name for name in names)
    
    # Rank 3 and 4 RDMs already active space tensors.
    return names

def rustIndex(rank: CumulantRank) -> str:
    """
    Emit the flat Rust index into an RDM tensor.

    Notation:
        gamma.data[p * n + q]

    Examples:
        rank 3 emits the index for gamma3.data[p,q,r,s,t,u].
    """
    return flatIndex(
        rustDataNames(rank),
        f"gamma{rank.rank}",
    )

def rustActiveIndexLocals(rank: CumulantRank, indent: str) -> list[str]:
    """
    Emit full-space active index conversions for rank 1 and rank 2.

    Notation:
        pp = active[p]

    Examples:
        rank 2 emits pp, qq, rr, ss.
    """
    if rank.rank > 2:
        return []

    lines = []

    for name in rank.upper + rank.lower:
        lines.append(f"{indent}let {name}{name} = active[{name}];")

    return lines

def rustLoopOpen(rank: CumulantRank) -> list[str]:
    """
    Emit nested Rust loops over all cumulant indices.

    Notation:
        for p in 0..n { ... }

    Examples:
        rank 2 emits loops over p, q, r, s.
    """
    lines = []

    for depth, name in enumerate(rank.upper + rank.lower):
        lines.append("    " * (1 + depth) + f"for {name} in 0..n {{")

    return lines

def rustLoopClose(rank: CumulantRank) -> list[str]:
    """
    Emit closing braces for nested Rust loops.

    Notation:
        }

    Examples:
        rank 2 emits four closing braces.
    """
    lines = []

    for depth in reversed(range(1, 1 + 2 * rank.rank)):
        lines.append("    " * depth + "}")

    return lines

def rustSet(rank: CumulantRank, value: str) -> str:
    """
    Emit one Rust cumulant tensor assignment.

    Notation:
        lambda.set(&[upper], &[lower], value)

    Examples:
        rank 3 emits lambda.set(&[p, q, r], &[s, t, u], value).
    """
    upper = ", ".join(rank.upper)
    lower = ", ".join(rank.lower)

    return f"lambda.set(&[{upper}], &[{lower}], {value});"

def rustUses(rank: CumulantRank) -> list[str]:
    """
    Emit Rust imports for one cumulant module.

    Notation:
        use ...

    Examples:
        rank 4 imports Cumulant1, Cumulant2, Cumulant3, and RDM4.
    """
    lines = [
        "use super::common::CumulantTensor;",
    ]

    if rank.rank >= 2:
        lines.append("use super::cumulants1::Cumulant1;")

    if rank.rank >= 3:
        lines.append("use super::cumulants2::Cumulant2;")

    if rank.rank >= 4:
        lines.append("use super::cumulants3::Cumulant3;")

    lines.extend([
        f"use crate::noci::rdm::RDM{rank.rank};",
        "use crate::noci::types::NOCIScalar;",
    ])

    return lines

def rustSignature(rank: CumulantRank) -> list[str]:
    """
    Emit the existing Rust cumulant function signature.

    Notation:
        pub(crate) fn cumulantsK(...)

    Examples:
        rank 2 keeps cumulants2(gamma2, lambda1, active).
    """
    if rank.rank == 1:
        return [
            "pub(crate) fn cumulants1<T: NOCIScalar>(",
            "    gamma1: &RDM1<T>,",
            "    active: &[usize],",
            ") -> Cumulant1<T> {",
        ]

    if rank.rank == 2:
        return [
            "pub(crate) fn cumulants2<T: NOCIScalar>(",
            "    gamma2: &RDM2<T>,",
            "    lambda1: &Cumulant1<T>,",
            "    active: &[usize],",
            ") -> Cumulant2<T> {",
        ]

    if rank.rank == 3:
        return [
            "pub(crate) fn cumulants3<T: NOCIScalar>(",
            "    gamma3: &RDM3<T>,",
            "    lambda1: &Cumulant1<T>,",
            "    lambda2: &Cumulant2<T>,",
            ") -> Cumulant3<T> {",
        ]

    if rank.rank == 4:
        return [
            "pub(crate) fn cumulants4<T: NOCIScalar>(",
            "    gamma4: &RDM4<T>,",
            "    lambda1: &Cumulant1<T>,",
            "    lambda2: &Cumulant2<T>,",
            "    lambda3: &Cumulant3<T>,",
            ") -> Cumulant4<T> {",
        ]

    raise ValueError(f"unsupported cumulant rank {rank.rank}")

def rustDoc(rank: CumulantRank) -> list[str]:
    """
    Emit the Rust doc comment for one cumulant builder.

    Notation:
        /// Build ...

    Examples:
        rank 4 documents gamma4, lambda1, lambda2, and lambda3.
    """
    names = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
    }

    lines = [
        f"/// Build the active-space spin-free {names[rank.rank]}-cumulant.",
        "/// # Arguments:",
        f"/// - `gamma{rank.rank}`: "
        f"{'Full-space' if rank.rank <= 2 else 'Active-space'} "
        f"spin-free {names[rank.rank]}-body RDM.",
    ]

    if rank.rank >= 2:
        lines.append("/// - `lambda1`: Active-space one-cumulant.")

    if rank.rank >= 3:
        lines.append("/// - `lambda2`: Active-space two-cumulant.")

    if rank.rank >= 4:
        lines.append("/// - `lambda3`: Active-space three-cumulant.")

    if rank.rank <= 2:
        lines.append("/// - `active`: Active orbital indices in the full NO basis.")

    lines.extend([
        "/// # Returns:",
        f"/// - `Cumulant{rank.rank}<T>`: Active-space "
        f"{names[rank.rank]}-cumulant.",
    ])

    return lines

def rustBodyIndent(rank: CumulantRank) -> str:
    """
    Return the indentation inside all generated index loops.

    Notation:
        one extra level inside all loops

    Examples:
        rank 4 returns nine Rust indentation levels.
    """
    return "    " * (1 + 2 * rank.rank)

def emitCumulant(rank: CumulantRank) -> str:
    """
    Emit one complete Rust cumulant module.

    Notation:
        cumulantsK.rs

    Examples:
        emitCumulant(RANKS[4]) emits cumulants4.rs.
    """
    disconnected = gammaDisconnectedExpr(rank)
    bodyIndent = rustBodyIndent(rank)
    expr = rustExpr(disconnected, bodyIndent)
    gammaName = f"gamma{rank.rank}"

    lines = [
        f"// noci/cumulants/cumulants{rank.rank}.rs",
        "// This file is generated by tools/wick/cumulants.py.",
        "// Do not edit generated cumulant expressions by hand.",
        "",
        *rustUses(rank),
        "",
        f"pub(crate) type Cumulant{rank.rank}<T> = CumulantTensor<T>;",
        "",
        *rustDoc(rank),
        *rustSignature(rank),
        f"    let n = {'active.len()' if rank.rank <= 2 else f'{gammaName}.n'};",
        f"    let mut lambda = CumulantTensor::zeros({rank.rank}, n);",
        "",
        *rustLoopOpen(rank),
        *rustActiveIndexLocals(rank, bodyIndent),
        f"{bodyIndent}let g{rank.rank}i = {rustIndex(rank)};",
    ]

    if rank.rank == 1:
        value = "gamma1.data[g1i]"
    else:
        lines.extend([
            "",
            f"{bodyIndent}let disconnected =",
            f"{expr};",
        ])
        value = f"gamma{rank.rank}.data[g{rank.rank}i] - disconnected"

    lines.extend([
        "",
        f"{bodyIndent}" + rustSet(rank, value),
        *rustLoopClose(rank),
        "",
        "    lambda",
        "}",
    ])

    return "\n".join(lines) + "\n"

def main() -> None:
    """
    Run the cumulant Rust generator.

    Notation:
        python tools/wick/cumulants.py --rank 4

    Examples:
        python tools/wick/cumulants.py --rank all --out-dir src/noci/cumulants
        regenerates cumulants1.rs through cumulants4.rs.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        choices = ("1", "2", "3", "4", "all"),
        default = "all",
        help = "cumulant rank to emit",
    )

    parser.add_argument(
        "--out-dir",
        default = None,
        help = "directory for generated cumulants*.rs files; stdout is used for one rank if omitted",
    )

    args = parser.parse_args()

    ranks = tuple(RANKS.values()) if args.rank == "all" else (RANKS[int(args.rank)],)

    if args.out_dir is None:
        if len(ranks) != 1:
            raise ValueError("--out-dir is required when --rank all")
        print(emitCumulant(ranks[0]), end = "")
        return

    outDir = Path(args.out_dir)
    outDir.mkdir(parents = True, exist_ok = True)

    for rank in ranks:
        (outDir / f"cumulants{rank.rank}.rs").write_text(
            emitCumulant(rank)
        )

if __name__ == "__main__":
    main()
