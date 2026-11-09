// qmc/shift.rs

/// Update the population-control shift from the logarithmic population growth, restored towards
/// the target population `N_*` with strength `\kappa`,
/// `E_s \leftarrow E_s - (\zeta/(A\delta\tau))[\ln(N_w/N_w^{\text{prev}}) + \kappa\ln(N_w^{\text{prev}}/N_*)]`,
/// once the population has first reached its target. When the shift enters the propagation only
/// through a death term, `\partial N_w/\partial E_s = A\delta\tau N_w`, so this is the damped Newton
/// step of the range-population controller of NOCIQMC.
/// # Arguments:
/// - `shift`: Population-control shift `E_s`.
/// - `reached`: Whether the population has reached its target, set on first arrival.
/// - `populations`: Current and previous population norms `(N_w, N_w^{\text{prev}})`.
/// - `control`: Target population `N_*`, damping `\zeta`, restoring strength `\kappa` and effective
///   time step `A\delta\tau`.
/// # Returns:
/// - `()`: Updates `shift` and `reached`.
pub(crate) fn update_shift(
    shift: &mut f64,
    reached: &mut bool,
    populations: (f64, f64),
    control: (f64, f64, f64, f64),
) {
    let (nw, previous) = populations;
    let (target, damping, restoring, dteff) = control;

    if !*reached && nw >= target {
        *reached = true;
    }

    if *reached {
        *shift -= (damping / dteff) * ((nw / previous).ln() + restoring * (previous / target).ln());
    }
}
