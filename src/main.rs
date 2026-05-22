use noci_rs::driver::{load_config, run};

fn main() {
    let config = load_config();
    run(config);
}
