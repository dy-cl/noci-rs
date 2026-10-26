// qmc/shift.rs

/// Update the population-control shift from the logarithmic population growth,
/// `E_s \leftarrow E_s - (\zeta/(A\delta\tau))\ln(N_w/N_w^{\text{prev}})`, once the population
/// has first reached its target.
/// # Arguments:
/// - `shift`: Population-control shift `E_s`.
/// - `reached`: Whether the population has reached its target, set on first arrival.
/// - `populations`: Current and previous population norms `(N_w, N_w^{\text{prev}})`.
/// - `control`: Target population, damping `\zeta` and effective time step `A\delta\tau`.
/// # Returns:
/// - `()`: Updates `shift` and `reached`.
pub(crate) fn update_shift(
    shift: &mut f64,
    reached: &mut bool,
    populations: (f64, f64),
    control: (f64, f64, f64),
) {
    let (nw, previous) = populations;
    let (target, damping, dteff) = control;

    if !*reached && nw >= target {
        *reached = true;
    }

    if *reached {
        *shift -= (damping / dteff) * (nw / previous).ln();
    }
}
