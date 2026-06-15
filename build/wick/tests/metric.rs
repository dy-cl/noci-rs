// tests/metric.rs

// tests/metric.rs

use wick_build::{overlap, target, wick};

/// Check one metric block against Appendix C.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `()`: Panics if the generated and target expressions differ.
fn check(name: &str) {
    let got = wick::canon(wick::eval(&overlap::block(name)));
    let want = wick::canon(target::block(name));

    assert_eq!(got, want, "Metric target mismatch for {name}");
}

/// Check C1 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C1 differs from Appendix C.
#[test]
fn c1() {
    check("C1");
}

/// Check C2 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C2 differs from Appendix C.
#[test]
fn c2() {
    check("C2");
}

/// Check C3 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C3 differs from Appendix C.
#[test]
fn c3() {
    check("C3");
}

/// Check C4 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C4 differs from Appendix C.
#[test]
fn c4() {
    check("C4");
}

/// Check C5 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C5 differs from Appendix C.
#[test]
fn c5() {
    check("C5");
}

/// Check C6 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C6 differs from Appendix C.
#[test]
fn c6() {
    check("C6");
}

/// Check C7 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C7 differs from Appendix C.
#[test]
fn c7() {
    check("C7");
}

/// Check C8 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C8 differs from Appendix C.
#[test]
fn c8() {
    check("C8");
}

/// Check C9 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C9 differs from Appendix C.
#[test]
fn c9() {
    check("C9");
}

/// Check C10 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C10 differs from Appendix C.
#[test]
fn c10() {
    check("C10");
}

/// Check C11 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C11 differs from Appendix C.
#[test]
fn c11() {
    check("C11");
}

/// Check C12 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C12 differs from Appendix C.
#[test]
fn c12() {
    check("C12");
}

/// Check C13 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C13 differs from Appendix C.
#[test]
fn c13() {
    check("C13");
}

/// Check C14 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C14 differs from Appendix C.
#[test]
fn c14() {
    check("C14");
}

/// Check C15 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C15 differs from Appendix C.
#[test]
fn c15() {
    check("C15");
}

/// Check C16 against Appendix C.
/// # Arguments:
/// - None.
/// # Returns:
/// - `()`: Panics if C16 differs from Appendix C.
#[test]
fn c16() {
    check("C16");
}
