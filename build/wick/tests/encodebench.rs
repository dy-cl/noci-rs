use std::time::Instant;

#[test]
#[ignore = "expensive encode benchmark"]
fn encoder2atoa() {
    let t = Instant::now();
    let x = wick_build::encode::residual_class(2, "AToA");

    eprintln!(
        "[bench]: R2(AToA) encoded terms: {}, elapsed: {:?}",
        x.terms.len(),
        t.elapsed()
    );
}
