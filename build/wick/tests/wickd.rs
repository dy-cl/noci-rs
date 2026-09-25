// tests/wickd.rs

// Standard library imports.
use std::fs;
use std::path::PathBuf;

// External crate imports.
use wick_build::so;

/// Check one spin-orbital residual against its Wick&D reference term by term.
/// The references in `tests/wickd` are produced by `tests/wickd/generate.py`.
/// # Arguments:
/// - `order`: Order in `T`.
/// - `class`: Spin-orbital excitation class name.
/// # Returns:
/// - `()`: Panics if any term or coefficient differs from the reference.
fn check(
    order: usize,
    class: &str,
) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/wickd")
        .join(format!("r{order}_{class}.txt"));
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let c = so::compare(order, class, &text);

    assert_eq!(
        (c.matching, c.mismatched, c.missing, c.extra),
        (c.reference, 0, 0, 0),
        "R{order}({class}) differs from Wick&D: {c:?}"
    );
}

/// Define one Wick&D comparison test per excitation class and order.
macro_rules! wickd_test {
    ($test:ident, $order:literal, $class:literal) => {
        /// Compare one generated spin-orbital residual with Wick&D.
        /// # Arguments:
        /// - None.
        /// # Returns:
        /// - `()`: Panics if the residual differs from the reference.
        #[test]
        fn $test() {
            check($order, $class);
        }
    };
    ($test:ident, $order:literal, $class:literal, ignore) => {
        /// Compare one slow generated spin-orbital residual with Wick&D.
        /// # Arguments:
        /// - None.
        /// # Returns:
        /// - `()`: Panics if the residual differs from the reference.
        #[test]
        #[ignore]
        fn $test() {
            check($order, $class);
        }
    };
}

wickd_test!(r0_ctoa, 0, "CToA");
wickd_test!(r0_atov, 0, "AToV");
wickd_test!(r0_atoa, 0, "AToA");
wickd_test!(r0_ctov, 0, "CToV");
wickd_test!(r0_catoav, 0, "CAToAV");
wickd_test!(r0_catovv, 0, "CAToVV");
wickd_test!(r0_cctoav, 0, "CCToAV");
wickd_test!(r0_cctoaa, 0, "CCToAA");
wickd_test!(r0_catoaa, 0, "CAToAA");
wickd_test!(r0_aatoav, 0, "AAToAV");
wickd_test!(r0_aatovv, 0, "AAToVV");
wickd_test!(r0_aatoaa, 0, "AAToAA");

wickd_test!(r1_ctoa, 1, "CToA");
wickd_test!(r1_atov, 1, "AToV");
wickd_test!(r1_atoa, 1, "AToA");
wickd_test!(r1_ctov, 1, "CToV");
wickd_test!(r1_catoav, 1, "CAToAV");
wickd_test!(r1_catovv, 1, "CAToVV");
wickd_test!(r1_cctoav, 1, "CCToAV");
wickd_test!(r1_cctoaa, 1, "CCToAA");
wickd_test!(r1_catoaa, 1, "CAToAA");
wickd_test!(r1_aatoav, 1, "AAToAV");
wickd_test!(r1_aatovv, 1, "AAToVV");
wickd_test!(r1_aatoaa, 1, "AAToAA");

wickd_test!(r2_ctoa, 2, "CToA");
wickd_test!(r2_atov, 2, "AToV");
wickd_test!(r2_atoa, 2, "AToA");
wickd_test!(r2_ctov, 2, "CToV");
wickd_test!(r2_catoav, 2, "CAToAV");
wickd_test!(r2_catovv, 2, "CAToVV");
wickd_test!(r2_cctoav, 2, "CCToAV");
wickd_test!(r2_cctoaa, 2, "CCToAA");
wickd_test!(r2_catoaa, 2, "CAToAA");
wickd_test!(r2_aatoav, 2, "AAToAV");
wickd_test!(r2_aatovv, 2, "AAToVV");
wickd_test!(r2_aatoaa, 2, "AAToAA", ignore);
