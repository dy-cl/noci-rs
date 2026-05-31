from core import *

def A(name: str) -> Idx:
    return Idx(name, Space.ACTIVE)

def C(name: str) -> Idx:
    return Idx(name, Space.CORE)

def V(name: str) -> Idx:
    return Idx(name, Space.VIRTUAL)

def L2(p: Idx, q: Idx, r: Idx, s: Idx) -> Expr:
    return tensor("Lambda2", (p, q), (r, s))

def checkSpinReplacement() -> None:
    p = A("p")
    q = A("q")
    r = A("r")
    s = A("s")

    assertEq(
        Spin.projectCumulant(
            1,
            (p,),
            (r,),
            (ALPHA,),
            (ALPHA,),
        ),
        scale(tensor("Gamma1", (p,), (r,)), Fraction(1, 2)),
        "Gamma1 alpha alpha",
    )

    assertEq(
        Spin.projectCumulant(
            1,
            (p,),
            (r,),
            (ALPHA,),
            (BETA,),
        ),
        zero(),
        "Gamma1 alpha beta",
    )

    assertEq(
        Spin.projectCumulant(
            2,
            (p, q),
            (r, s),
            (ALPHA, ALPHA),
            (ALPHA, ALPHA),
        ),
        add(
            scale(L2(p, q, r, s), Fraction(1, 6)),
            scale(L2(p, q, s, r), Fraction(-1, 6)),
        ),
        "Lambda2 alpha alpha / alpha alpha",
    )

    assertEq(
        Spin.projectCumulant(
            2,
            (p, q),
            (r, s),
            (ALPHA, BETA),
            (ALPHA, BETA),
        ),
        add(
            scale(L2(p, q, r, s), Fraction(2, 6)),
            scale(L2(p, q, s, r), Fraction(1, 6)),
        ),
        "Lambda2 alpha beta / alpha beta",
    )

    assertEq(
        Spin.projectCumulant(
            2,
            (p, q),
            (r, s),
            (ALPHA, BETA),
            (BETA, ALPHA),
        ),
        add(
            scale(L2(p, q, r, s), Fraction(-1, 6)),
            scale(L2(p, q, s, r), Fraction(-2, 6)),
        ),
        "Lambda2 alpha beta / beta alpha",
    )

def checkWick() -> None:
    ref = Ref()
    wick = Wick(ref)

    u = A("u")
    v = A("v")

    assertEq(
        wick.eval(Product((tau1(u, v, 0),))),
        zero(),
        "single GNO group expectation",
    )

def assertEq(got, want, label: str) -> None:
    if got != want:
        print(f"===== {label} got =====")
        print(got)
        print(f"===== {label} want =====")
        print(want)
        raise SystemExit(1)

def main() -> None:
    i = C("i")
    j = C("j")
    a = V("a")
    b = V("b")
    u = A("u")
    v = A("v")

    assertEq(
        anticommutator(
            Op("annihilate", u, ALPHA, 0),
            Op("create", v, ALPHA, 1),
        ),
        delta(u, v),
        "CAR same spin",
    )

    assertEq(
        anticommutator(
            Op("annihilate", u, ALPHA, 0),
            Op("create", v, BETA, 1),
        ),
        zero(),
        "CAR opposite spin",
    )

    ref = Ref()

    assertEq(
        ref.frozenPair(
            Op("create", i, ALPHA, 0),
            Op("annihilate", j, ALPHA, 1),
        ),
        delta(i, j),
        "core occupied pair",
    )

    assertEq(
        ref.frozenPair(
            Op("annihilate", a, ALPHA, 0),
            Op("create", b, ALPHA, 1),
        ),
        delta(a, b),
        "virtual empty pair",
    )

    assertEq(
        len(groupE1(u, v, 0).strings),
        2,
        "E1 spin expansion length",
    )

    assertEq(
        len(groupE2(u, v, i, j, 0).strings),
        4,
        "E2 spin expansion length",
    )

    checkSpinReplacement()
    checkWick()
    print("Tests pass.")

if __name__ == "__main__":
    main()
