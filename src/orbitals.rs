// orbitals.rs

use std::sync::Arc;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::AoData;
use crate::DetState;
use crate::maths::{ERIScalar, adjoint, loewdin_x, real2_as};
use crate::noci::{NOCIData, NOCIScalar, build_noci_s, noci_density, occ_coeffs};

/// Stores a common orthonormal natural-orbital basis and its occupation partition.
pub(crate) struct NOCINaturalOrbitals {
    /// AO coefficients of the NOCI natural orbital basis.
    pub c: Array2<f64>,
    /// Spin-free natural occupations, sorted in descending order.
    pub occupations: Array1<f64>,
    /// Orbitals with occupation close to two.
    pub core: Vec<usize>,
    /// Orbitals with fractional occupation.
    pub active: Vec<usize>,
    /// Orbitals with occupation close to zero.
    pub _virtuals: Vec<usize>,
}

/// Build NOCI natural orbitals from the spin-free NOCI one-body RDM.
/// # Arguments:
/// - `data`: Shared data required for NOCI matrix-element evaluation.
/// - `coeffs`: NOCI coefficient vector.
/// - `ctol`: Occupation tolerance for assigning core orbitals.
/// - `vitol`: Occupation tolerance for assigning virtual orbitals.
/// # Returns:
/// - `NaturalOrbitalBasis`: Natural orbitals and core/active/virtual partition.
pub(crate) fn noci_natural_orbitals(
    data: &NOCIData<'_, f64>,
    coeffs: &Array1<f64>,
    ctol: f64,
    vtol: f64,
) -> NOCINaturalOrbitals {
    // Get overlap matrix of states in NOCI basis.
    let (s, _) = build_noci_s(data, data.basis, data.basis, true);
    let norm = coeffs.dot(&s.dot(coeffs));

    // Get AO density matrix of states in NOCI basis and normalise.
    let (da, db) = noci_density(data.ao, data.basis, coeffs, data.tol);
    let mut d = da + db;
    d.mapv_inplace(|x| x / norm);

    // Build symmetric orthogonaliser and transform AO NOCI density into the orthonormal basis.
    let x = loewdin_x(&data.ao.s, false, data.tol);
    let gamma = x.t().dot(&data.ao.s).dot(&d).dot(&data.ao.s).dot(&x);

    // Diagonalise the orthonormal RDM \Gamma for occupancies and orbitals.
    let (occs, u) = gamma.eigh(UPLO::Lower).unwrap();

    // Sort orbitals in descending order by occupation.
    let mut order: Vec<usize> = (0..occs.len()).collect();
    order.sort_by(|&i, &j| occs[j].partial_cmp(&occs[i]).unwrap());

    let norb = occs.len();
    let mut occupations = Array1::<f64>::zeros(norb);
    let mut orbitals = Array2::<f64>::zeros((norb, norb));

    // Apply the descending sort to occupations and eigenvectors (orbitals).
    for (new, &old) in order.iter().enumerate() {
        occupations[new] = occs[old];
        orbitals.column_mut(new).assign(&u.column(old));
    }

    // Transform natural NOCI orbitals back to AO coefficients.
    let c = x.dot(&orbitals);
    let (core, active, virtuals) = partition_natural_occupations(&occupations, ctol, vtol);

    NOCINaturalOrbitals {
        c,
        occupations,
        core,
        active,
        _virtuals: virtuals,
    }
}

/// Partition spin-free natural occupations into core, active and virtual spaces.
/// # Arguments:
/// - `occupations`: Spin-free natural occupations.
/// - `ctol`: Occupation tolerance for assigning core orbitals.
/// - `vtol`: Occupation tolerance for assigning virtual orbitals.
/// # Returns:
/// - `(Vec<usize>, Vec<usize>, Vec<usize>)`: Core, active and virtual orbital indices.
fn partition_natural_occupations(
    occupations: &Array1<f64>,
    ctol: f64,
    vtol: f64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut core = Vec::new();
    let mut active = Vec::new();
    let mut virtuals = Vec::new();

    for (p, &n) in occupations.iter().enumerate() {
        if n > 2.0 - ctol {
            core.push(p);
        } else if n > vtol {
            active.push(p);
        } else {
            virtuals.push(p);
        }
    }

    (core, active, virtuals)
}

/// Transform a NOCI determinant basis into a supplied orbital basis.
/// # Arguments:
/// - `basis`: Determinants to transform.
/// - `c`: AO coefficients of the orthonormal orbital basis.
/// - `s`: AO overlap matrix.
/// # Returns:
/// - `Vec<DetState<T>>`: Determinants represented in the supplied orbital basis.
pub(crate) fn transform_noci_basis<T: NOCIScalar>(
    basis: &[DetState<T>],
    c: &Array2<T>,
    s: &Array2<f64>,
) -> Vec<DetState<T>> {
    let s = real2_as::<T>(s);
    let cdag_s = adjoint(c).dot(&s);

    basis
        .iter()
        .map(|det| {
            let ca = cdag_s.dot(det.ca.as_ref());
            let cb = cdag_s.dot(det.cb.as_ref());

            let ca_occ = occ_coeffs(&ca, det.oa);
            let cb_occ = occ_coeffs(&cb, det.ob);
            let da = ca_occ.dot(&adjoint(&ca_occ));
            let db = cb_occ.dot(&adjoint(&cb_occ));

            let mut out = det.clone();
            out.ca = Arc::new(ca);
            out.cb = Arc::new(cb);
            out.da = Arc::new(da);
            out.db = Arc::new(db);
            out
        })
        .collect()
}

/// Transform AO data into a supplied orbital basis.
/// # Arguments:
/// - `ao`: Original AO-basis one- and two-electron data.
/// - `c`: AO coefficients of the target orbital basis.
/// # Returns:
/// - `AoData`: One- and two-electron data represented in the target basis.
pub(crate) fn transform_ao_data(
    ao: &AoData,
    c: &Array2<f64>,
) -> AoData {
    let s = c.t().dot(&ao.s).dot(c);
    let h = c.t().dot(&ao.h).dot(c);
    let dm = c.t().dot(&ao.s).dot(&ao.dm).dot(&ao.s).dot(c);

    let nmo = c.ncols();
    let mut eri_coul = ndarray::Array4::<f64>::zeros((nmo, nmo, nmo, nmo));
    let mut scratch = f64::new_eri_ao2mo_scratch(&ao.eri_coul, nmo, nmo, nmo, nmo);
    f64::eri_ao2mo_hermitian_into(&ao.eri_coul, c, c, c, c, eri_coul.view_mut(), &mut scratch);
    let mut eri_asym = ndarray::Array4::<f64>::zeros((nmo, nmo, nmo, nmo));
    let mut scratch = f64::new_eri_ao2mo_scratch(&ao.eri_asym, nmo, nmo, nmo, nmo);
    f64::eri_ao2mo_hermitian_into(&ao.eri_asym, c, c, c, c, eri_asym.view_mut(), &mut scratch);

    let labels = (0..c.ncols()).map(|i| format!("Orbital {}", i)).collect();

    AoData {
        s,
        h,
        dm,
        eri_coul,
        eri_asym,
        enuc: ao.enuc,
        n: c.ncols(),
        nelec: ao.nelec.clone(),
        labels,
        e_fci: ao.e_fci,
    }
}

/// Print NOCI natural orbital occupations and partitions.
/// # Arguments:
/// - `title`: Title for the natural orbital table.
/// - `no`: NOCI natural orbital basis and occupation partition.
/// # Returns:
/// - `()`: Prints natural orbital occupations and core/active/virtual labels.
pub(crate) fn print_noci_natural_orbitals(
    title: &str,
    no: &NOCINaturalOrbitals,
) {
    println!("{}", "-".repeat(100));
    println!("{title}:");
    println!("{:^5} {:^14} {:^10}", "NO", "Occ", "Space");

    for (i, &n) in no.occupations.iter().enumerate() {
        let space = if no.core.contains(&i) {
            "Core"
        } else if no.active.contains(&i) {
            "Active"
        } else {
            "Virtual"
        };

        println!("{:^5} {:^14.8} {:^10}", i, n, space);
    }
}
