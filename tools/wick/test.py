from core import *

def A(name: str) -> Idx:
    return Idx(name, Space.ACTIVE)

def C(name: str) -> Idx:
    return Idx(name, Space.CORE)

def V(name: str) -> Idx:
    return Idx(name, Space.VIRTUAL)

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

    print("Tests pass.")

if __name__ == "__main__":
    main()
