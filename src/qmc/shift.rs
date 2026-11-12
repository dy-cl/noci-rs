// qmc/shift.rs

/// Take the damped Newton step of the population-control shift once the population has first
/// reached its target `N_*`, restored towards it with strength `\kappa`,
/// `E_s \leftarrow E_s - s[\ln(N_w/N_w^{\text{prev}}) + \kappa\ln(N_w^{\text{prev}}/N_*)]`,
/// with the Newton scale `s = \zeta N_w(\partial N_w/\partial E_s)^{-1}`. A propagator whose shift
/// enters only through a death term has `\partial N_w/\partial E_s = A\delta\tau N_w`, so
/// `s = \zeta/(A\delta\tau)`; a range propagator takes `\partial N_w/\partial E_s` from its
/// propagated tangent.
/// # Arguments:
/// - `shift`: Population-control shift `E_s`.
/// - `reached`: Whether the population has reached its target, set on first arrival.
/// - `populations`: Current and previous population norms `(N_w, N_w^{\text{prev}})`.
/// - `control`: Target population `N_*` and restoring strength `\kappa`.
/// - `scale`: Newton scale `s`, or `None` when the Jacobian is unusable and the shift is kept.
/// # Returns:
/// - `()`: Updates `shift` and `reached`; a non-finite step leaves the shift unchanged.
pub(crate) fn update_shift(
    shift: &mut f64,
    reached: &mut bool,
    populations: (f64, f64),
    control: (f64, f64),
    scale: Option<f64>,
) {
    let (nw, previous) = populations;
    let (target, restoring) = control;

    if !*reached && nw >= target {
        *reached = true;
    }

    if let Some(scale) = scale.filter(|_| *reached) {
        let next = *shift - scale * ((nw / previous).ln() + restoring * (previous / target).ln());
        if next.is_finite() {
            *shift = next;
        }
    }
}
