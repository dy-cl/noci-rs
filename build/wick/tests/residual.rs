// tests/residual.rs

use std::collections::BTreeMap;

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

/// Generate residual chunks.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation class name.
/// # Returns:
/// - `BTreeMap<String, _>`: Canonical generated chunks keyed by chunk label.
fn generated(
    order: Order,
    name: &str,
) -> BTreeMap<String, Vec<wick_build::ir::Term>> {
    let mut out = BTreeMap::new();

    match order {
        Order::R0 => residual::r0(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
        Order::R1 => residual::r1(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
        Order::R2 => residual::r2(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
    }

    out
}

/// Generate target residual chunks.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation class name.
/// # Returns:
/// - `BTreeMap<String, _>`: Canonical target chunks keyed by chunk label.
fn expected(
    order: Order,
    name: &str,
) -> BTreeMap<String, Vec<wick_build::ir::Term>> {
    let mut out = BTreeMap::new();

    match order {
        Order::R0 => target::r0(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
        Order::R1 => target::r1(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
        Order::R2 => target::r2(name, |k, e| {
            out.insert(k, wick::canon(e));
        }),
    }

    out
}

/// Check one residual class against target chunks.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation class name.
/// # Returns:
/// - `()`: Panics if generated and target chunks differ.
fn check(
    order: Order,
    name: &str,
) {
    let got = generated(order, name);
    let want = expected(order, name);

    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{} chunk-key mismatch for {name}",
        order.label(),
    );

    for (key, got_expr) in got {
        let want_expr = want.get(&key).unwrap();

        assert_eq!(
            &got_expr,
            want_expr,
            "{} target mismatch for {name}, chunk {key}",
            order.label(),
        );
    }
}

/// Check that one residual class can be generated without storing chunks.
/// # Arguments:
/// - `order`: Residual order.
/// - `name`: Excitation class name.
/// # Returns:
/// - `()`: Panics only if generation itself fails.
fn sink(
    order: Order,
    name: &str,
) {
    let mut chunks = 0usize;
    let mut terms = 0usize;

    match order {
        Order::R0 => residual::r0(name, |k, e| {
            chunks += 1;
            terms += e.len();
            eprintln!(
                "[sink]: Order: {}, Class: {name}, Chunk: {k}, Terms: {}, Total Chunks: {chunks}, Total Terms: {terms}",
                order.label(),
                e.len(),
            );
            drop(e);
        }),
        Order::R1 => residual::r1(name, |k, e| {
            chunks += 1;
            terms += e.len();
            eprintln!(
                "[sink]: Order: {}, Class: {name}, Chunk: {k}, Terms: {}, Total Chunks: {chunks}, Total Terms: {terms}",
                order.label(),
                e.len(),
            );
            drop(e);
        }),
        Order::R2 => residual::r2(name, |k, e| {
            chunks += 1;
            terms += e.len();
            eprintln!(
                "[sink]: Order: {}, Class: {name}, Chunk: {k}, Terms: {}, Total Chunks: {chunks}, Total Terms: {terms}",
                order.label(),
                e.len(),
            );
            drop(e);
        }),
    }

    eprintln!(
        "[sink]: Order: {}, Class: {name}, Done: true, Chunks: {chunks}, Terms: {terms}",
        order.label(),
    );
}

macro_rules! residual_test {
    ($test:ident, $order:ident, $class:literal) => {
        #[test]
        fn $test() {
            check(Order::$order, $class);
        }
    };
}

macro_rules! residual_sink_test {
    ($test:ident, $order:ident, $class:literal) => {
        #[test]
        #[ignore]
        fn $test() {
            sink(Order::$order, $class);
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

residual_sink_test!(r1_ctoa, R1, "CToA");
residual_sink_test!(r1_atoa, R1, "AToA");
residual_sink_test!(r1_atov, R1, "AToV");
residual_sink_test!(r1_catoav, R1, "CAToAV");
residual_sink_test!(r1_catova, R1, "CAToVA");
residual_sink_test!(r1_catovv, R1, "CAToVV");
residual_sink_test!(r1_cctoav, R1, "CCToAV");
residual_sink_test!(r1_cctoaa, R1, "CCToAA");
residual_sink_test!(r1_catoaa, R1, "CAToAA");
residual_sink_test!(r1_aatoav, R1, "AAToAV");
residual_sink_test!(r1_aatovv, R1, "AAToVV");
residual_sink_test!(r1_aatoaa, R1, "AAToAA");

residual_sink_test!(r2_ctoa, R2, "CToA");
residual_sink_test!(r2_atoa, R2, "AToA");
residual_sink_test!(r2_atov, R2, "AToV");
residual_sink_test!(r2_catoav, R2, "CAToAV");
residual_sink_test!(r2_catova, R2, "CAToVA");
residual_sink_test!(r2_catovv, R2, "CAToVV");
residual_sink_test!(r2_cctoav, R2, "CCToAV");
residual_sink_test!(r2_cctoaa, R2, "CCToAA");
residual_sink_test!(r2_catoaa, R2, "CAToAA");
residual_sink_test!(r2_aatoav, R2, "AAToAV");
residual_sink_test!(r2_aatovv, R2, "AAToVV");
residual_sink_test!(r2_aatoaa, R2, "AAToAA");
