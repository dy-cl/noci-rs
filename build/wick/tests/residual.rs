// tests/residual.rs

use wick_build::{residual, target, wick};

/// Residual order.
#[derive(Clone, Copy)]
enum Order {
    /// Zeroth-order residual.
    R0,
    /// First-order residual.
    R1,
    /// Second-order residual.
    R2,
}

impl Order {
    /// Return the residual-order label.
    /// # Arguments:
    /// - `self`: Residual order.
    /// # Returns:
    /// - `&'static str`: Residual-order label.
    fn label(self) -> &'static str {
        match self {
            Order::R0 => "R0",
            Order::R1 => "R1",
            Order::R2 => "R2",
        }
    }
}

/// Check one residual class.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation class name.
/// # Returns:
/// - `()`: Panics if generated and target expressions differ.
fn check(order: Order, name: &str) {
    let (got, want) = match order {
        Order::R0 => (
            wick::canon(residual::r0(name)),
            wick::canon(target::r0(name)),
        ),
        Order::R1 => (
            wick::canon(residual::r1(name)),
            wick::canon(target::r1(name)),
        ),
        Order::R2 => (
            wick::canon(residual::r2(name)),
            wick::canon(target::r2(name)),
        ),
    };

    assert_eq!(got, want, "{} target mismatch for {name}", order.label());
}

macro_rules! residual_test {
    ($test:ident, $order:ident, $class:literal) => {
        #[test]
        fn $test() {
            check(Order::$order, $class);
        }
    };
}

macro_rules! residual_ignored_test {
    ($test:ident, $order:ident, $class:literal) => {
        #[test]
        #[ignore]
        fn $test() {
            check(Order::$order, $class);
        }
    };
}

residual_test!(r0_ctoa, R0, "CToA");
residual_test!(r0_atoa, R0, "AToA");
residual_test!(r0_atov, R0, "AToV");
residual_test!(r0_catoav, R0, "CAToAV");
residual_test!(r0_catova, R0, "CAToVA");
residual_test!(r0_catovv, R0, "CAToVV");
residual_test!(r0_cctoav, R0, "CCToAV");
residual_test!(r0_cctoaa, R0, "CCToAA");
residual_test!(r0_catoaa, R0, "CAToAA");
residual_test!(r0_aatoav, R0, "AAToAV");
residual_test!(r0_aatovv, R0, "AAToVV");
residual_test!(r0_aatoaa, R0, "AAToAA");

residual_test!(r1_ctoa, R1, "CToA");
residual_test!(r1_atoa, R1, "AToA");
residual_test!(r1_atov, R1, "AToV");
residual_test!(r1_catoav, R1, "CAToAV");
residual_test!(r1_catova, R1, "CAToVA");
residual_test!(r1_catovv, R1, "CAToVV");
residual_test!(r1_cctoav, R1, "CCToAV");
residual_test!(r1_cctoaa, R1, "CCToAA");
residual_test!(r1_catoaa, R1, "CAToAA");
residual_test!(r1_aatoav, R1, "AAToAV");
residual_test!(r1_aatovv, R1, "AAToVV");
residual_test!(r1_aatoaa, R1, "AAToAA");

residual_test!(r2_ctoa, R2, "CToA");
residual_ignored_test!(r2_atoa, R2, "AToA");
residual_ignored_test!(r2_atov, R2, "AToV");
residual_ignored_test!(r2_catoav, R2, "CAToAV");
residual_ignored_test!(r2_catova, R2, "CAToVA");
residual_ignored_test!(r2_catovv, R2, "CAToVV");
residual_ignored_test!(r2_cctoav, R2, "CCToAV");
residual_ignored_test!(r2_cctoaa, R2, "CCToAA");
residual_ignored_test!(r2_catoaa, R2, "CAToAA");
residual_ignored_test!(r2_aatoav, R2, "AAToAV");
residual_ignored_test!(r2_aatovv, R2, "AAToVV");
residual_ignored_test!(r2_aatoaa, R2, "AAToAA");
