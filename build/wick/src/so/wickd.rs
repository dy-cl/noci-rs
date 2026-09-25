// so/wickd.rs

// External crate imports.
use num_rational::Ratio;
use smallvec::SmallVec;

// Parent/sibling imports.
use super::{Expr, Kind, Space, Tensor, Term};

/// Parse Wick&D's text output into a canonical spin-orbital expression.
/// Each non-empty line is one term such as `-1/2 R^{a0}_{o0} f^{o0}_{o1} lambda2^{a0,a1}_{a2,a3}`:
/// an optional rational coefficient followed by tensors `label^{upper}_{lower}` whose indices are
/// named by space (`o`, `a`, `v`) and number.
/// # Arguments:
/// - `text`: Wick&D expression text.
/// # Returns:
/// - `Expr`: Canonically combined expression.
pub(crate) fn parse(text: &str) -> Expr {
    let mut acc = Expr::default();

    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        // Split off the leading coefficient, if any.
        let (coeff, rest) = coefficient(line);
        let mut names = Vec::<String>::new();
        let mut spaces = Vec::new();
        let mut tensors = Vec::new();

        for token in rest.split_whitespace() {
            let (label, rest) = token
                .split_once("^{")
                .unwrap_or_else(|| panic!("bad tensor {token}"));
            let (upper, lower) = rest
                .split_once("}_{")
                .unwrap_or_else(|| panic!("bad tensor {token}"));
            let lower = lower.trim_end_matches('}');

            let mut ids = |xs: &str| {
                xs.split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| {
                        let pos = names.iter().position(|n| n == x).unwrap_or_else(|| {
                            names.push(x.to_string());
                            spaces.push(space(x));
                            names.len() - 1
                        });
                        pos as u16
                    })
                    .collect::<SmallVec<[u16; 4]>>()
            };
            let upper = ids(upper);
            let lower = ids(lower);

            tensors.push(Tensor {
                kind: kind(label, upper.len()),
                upper,
                lower,
            });
        }

        super::add(
            &mut acc,
            &Term {
                coeff,
                spaces,
                tensors,
            },
        );
    }

    acc
}

/// Split one term line into its coefficient and tensor text.
/// # Arguments:
/// - `line`: Term line.
/// # Returns:
/// - `(Ratio<i64>, &str)`: Coefficient and remaining tensor text.
fn coefficient(line: &str) -> (Ratio<i64>, &str) {
    let (first, rest) = line.split_once(' ').unwrap_or((line, ""));

    // A leading bare sign or rational is a coefficient; otherwise the coefficient is one.
    if first == "+" || first == "-" {
        let sign = if first == "-" { -1 } else { 1 };
        return (Ratio::from_integer(sign), rest);
    }
    if !first.contains('^') {
        let (num, den) = first.split_once('/').unwrap_or((first, "1"));
        let num = num
            .trim_start_matches('+')
            .parse::<i64>()
            .unwrap_or_else(|e| panic!("bad coefficient {first}: {e}"));
        let den = den
            .parse::<i64>()
            .unwrap_or_else(|e| panic!("bad coefficient {first}: {e}"));
        return (Ratio::new(num, den), rest);
    }

    let sign = if first.starts_with('-') { -1 } else { 1 };
    (
        Ratio::from_integer(sign),
        line.trim_start_matches(['+', '-']),
    )
}

/// Return the orbital space of one Wick&D index name.
/// # Arguments:
/// - `name`: Index name such as `a3`.
/// # Returns:
/// - `Space`: Orbital space.
fn space(name: &str) -> Space {
    match name.as_bytes()[0] {
        b'o' => Space::Core,
        b'a' => Space::Active,
        b'v' => Space::Virtual,
        _ => panic!("unknown index space {name}"),
    }
}

/// Return the tensor kind of one Wick&D label.
/// # Arguments:
/// - `label`: Tensor label.
/// - `rank`: Number of upper indices.
/// # Returns:
/// - `Kind`: Spin-orbital tensor kind.
fn kind(
    label: &str,
    rank: usize,
) -> Kind {
    match (label, rank) {
        ("R", _) => Kind::Bra,
        ("f", _) => Kind::Fock,
        ("v", _) => Kind::Eri,
        ("t", 1) => Kind::T1,
        ("t", _) => Kind::T2,
        ("gamma1", _) => Kind::Gamma,
        ("eta1", _) => Kind::Eta,
        ("lambda2", _) => Kind::Lambda2,
        ("lambda3", _) => Kind::Lambda3,
        ("lambda4", _) => Kind::Lambda4,
        _ => panic!("unknown Wick&D tensor {label}"),
    }
}
