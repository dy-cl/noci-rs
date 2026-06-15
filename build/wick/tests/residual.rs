// tests/residual.rs

use wick_build::{residual, target, wick};

/// Check one zeroth-order residual class.
/// # Arguments:
/// - `name`: Excitation class name.
/// # Returns:
/// - `()`: Panics if generated and target expressions differ.
fn check(name: &str) {
    let got = wick::canon(residual::r0(name));
    let want = wick::canon(target::r0(name));

    assert_eq!(got, want, "R0 target mismatch for {name}");
}

macro_rules! r0_test {
    ($test:ident, $class:literal) => {
        #[test]
        fn $test() {
            check($class);
        }
    };
}

r0_test!(r0_ctoa, "CToA");
r0_test!(r0_atoa, "AToA");
r0_test!(r0_atov, "AToV");
r0_test!(r0_catoav, "CAToAV");
r0_test!(r0_catova, "CAToVA");
r0_test!(r0_catovv, "CAToVV");
r0_test!(r0_cctoav, "CCToAV");
r0_test!(r0_cctoaa, "CCToAA");
r0_test!(r0_catoaa, "CAToAA");
r0_test!(r0_aatoav, "AAToAV");
r0_test!(r0_aatovv, "AAToVV");
r0_test!(r0_aatoaa, "AAToAA");
