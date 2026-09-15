// basis/excitation.rs

/// Calculate fermionic sign associated with applying a set of creation and annihilation operators to a determinant described by a bitstring.
/// # Arguments:
/// - `occ`: Occupancy bitstring.
/// - `holes`: Annihilation operators indices.
/// - `parts`: Creation operator indices.
/// # Returns
/// - `f64`: Fermionic phase factor.
#[inline(always)]
pub fn excitation_phase(
    mut occ: u128,
    holes: &[usize],
    parts: &[usize],
) -> f64 {
    /// Determine whether the number of occupied orbitals below orbital index `p` is odd.
    /// # Arguments:
    /// - `bits`: Occupancy bitstring.
    /// - `p`: Orbital index.
    /// # Returns:
    /// - `bool`: `true` if the number of occupied orbitals with index less than `p` is odd, otherwise `false`.
    #[inline(always)]
    fn below(
        bits: u128,
        p: usize,
    ) -> bool {
        if p == 0 {
            false
        } else {
            ((bits & ((1u128 << p) - 1)).count_ones() & 1) != 0
        }
    }

    let mut odd = false;

    for &i in holes.iter().rev() {
        odd ^= below(occ, i);
        occ &= !(1u128 << i);
    }

    for (k, &a) in parts.iter().enumerate() {
        odd ^= below(occ, a);
        if k + 1 != parts.len() {
            occ |= 1u128 << a;
        }
    }
    if odd { -1.0 } else { 1.0 }
}

/// Calculate fermionic sign associated with applying stored excitation bit masks to a determinant.
/// # Arguments:
/// - `occ`: Occupancy bitstring.
/// - `holes`: Annihilation operator bit mask.
/// - `parts`: Creation operator bit mask.
/// # Returns
/// - `f64`: Fermionic phase factor.
#[inline(always)]
pub(crate) fn excitation_phase_bits(
    mut occ: u128,
    mut holes: u128,
    mut parts: u128,
) -> f64 {
    #[inline(always)]
    fn below(
        bits: u128,
        p: usize,
    ) -> bool {
        if p == 0 {
            false
        } else {
            ((bits & ((1u128 << p) - 1)).count_ones() & 1) != 0
        }
    }

    let mut odd = false;

    while holes != 0 {
        let i = 127 - holes.leading_zeros() as usize;
        odd ^= below(occ, i);
        occ &= !(1u128 << i);
        holes &= !(1u128 << i);
    }

    while parts != 0 {
        let a = parts.trailing_zeros() as usize;
        odd ^= below(occ, a);
        parts &= parts - 1;
        if parts != 0 {
            occ |= 1u128 << a;
        }
    }

    if odd { -1.0 } else { 1.0 }
}

/// Undo a spin-specific excitation from a child occupation bitstring.
/// # Arguments
/// - `occ`: Child occupation bitstring.
/// - `holes`: Orbitals removed from the parent determinant.
/// - `parts`: Orbitals added to the parent determinant.
/// # Returns
/// - `u128`: Reconstructed parent occupation bitstring.
pub(crate) fn undo_excitation(
    occ: u128,
    holes: u128,
    parts: u128,
) -> u128 {
    (occ & !parts) | holes
}

/// Construct the excitation mapping one occupation bitstring to another.
/// # Arguments
/// - `parent`: Parent occupation bitstring.
/// - `child`: Child occupation bitstring.
/// # Returns
/// - `(u128, u128)`: Hole and particle orbital masks.
#[inline(always)]
pub(crate) fn excitation_between(
    parent: u128,
    child: u128,
) -> (u128, u128) {
    (parent & !child, child & !parent)
}
