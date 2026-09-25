// tests/metric.rs

// External crate imports.
use wick_build::target;

/// Define one Appendix C comparison test per metric block.
macro_rules! metric_test {
    ($test:ident, $name:literal) => {
        /// Compare one generated metric block with Appendix C.
        /// # Arguments:
        /// - None.
        /// # Returns:
        /// - `()`: Panics if the block differs from Appendix C.
        #[test]
        fn $test() {
            assert!(target::check($name), "{} differs from Appendix C", $name);
        }
    };
}

metric_test!(c1, "C1");
metric_test!(c2, "C2");
metric_test!(c3, "C3");
metric_test!(c4, "C4");
metric_test!(c5, "C5");
metric_test!(c6, "C6");
metric_test!(c7, "C7");
metric_test!(c8, "C8");
metric_test!(c9, "C9");
metric_test!(c10, "C10");
metric_test!(c11, "C11");
metric_test!(c12, "C12");
metric_test!(c13, "C13");
metric_test!(c14, "C14");
metric_test!(c15, "C15");
metric_test!(c16, "C16");
