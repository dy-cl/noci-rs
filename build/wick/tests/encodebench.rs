#[test]
#[ignore]
fn encoder2atoa() {
    wick_build::timers::reset_all();

    let start = std::time::Instant::now();
    let terms = wick_build::encode::residual_class(2, "AToA");
    let elapsed = start.elapsed();

    wick_build::timers::print_all();

    println!(
        "[bench]: R2(AToA) encoded terms: {}, elapsed: {:?}",
        terms.terms.len(),
        elapsed
    );
}
