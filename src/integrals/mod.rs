// integrals/mod.rs
//! In-memory atomic-orbital integral generation.

// Standard library imports.
use std::collections::HashMap;

// External crate imports.
use libcint::prelude::{CInt, CIntMol};
use ndarray::{Array1, Array2, Array4, Axis, s};

// Crate-root imports.
use crate::AoData;
use crate::maths::{general_evp_x, loewdin_x};

/// Evaluate a two-index integral in row-major AO order.
/// # Arguments:
/// - `cint`: Integral environment.
/// - `intor`: libcint integral name.
/// # Returns:
/// - `Array2<f64>`: Integral matrix with shape `(nao, nao)`.
fn integral2(
    cint: &CInt,
    intor: &str,
) -> Array2<f64> {
    let nao = cint.nao();
    let (out, _shape): (Vec<f64>, Vec<usize>) = cint.integrate_row_major(intor, None, None).into();

    Array2::from_shape_vec((nao, nao), out).unwrap()
}

/// Evaluate a four-index integral in row-major AO order.
/// # Arguments:
/// - `cint`: Integral environment.
/// - `intor`: libcint integral name.
/// # Returns:
/// - `Array4<f64>`: Integral tensor with shape `(nao, nao, nao, nao)`.
fn integral4(
    cint: &CInt,
    intor: &str,
) -> Array4<f64> {
    let nao = cint.nao();
    let (out, _shape): (Vec<f64>, Vec<usize>) = cint.integrate_row_major(intor, None, None).into();

    Array4::from_shape_vec((nao, nao, nao, nao), out).unwrap()
}

/// Antisymmetrise the ERIs as `(ab||cd) = (ab|cd) - (ac|bd)`.
/// # Arguments:
/// - `eri_coul`: Coulomb ERIs.
/// # Returns:
/// - `Array4<f64>`: Antisymmetrised ERIs.
fn antisymmetrise(eri_coul: &Array4<f64>) -> Array4<f64> {
    let eri_exch = eri_coul.view().permuted_axes([0, 2, 1, 3]).to_owned();

    eri_coul - &eri_exch
}

/// Compute nuclear repulsion energy `E_nuc = sum_{A<B} Z_A Z_B / R_AB`.
/// # Arguments:
/// - `cint`: Integral environment.
/// # Returns:
/// - `f64`: Nuclear repulsion energy.
fn nuclear_repulsion(cint: &CInt) -> f64 {
    let coords = cint.atom_coords();
    let charges = cint.atom_charges();
    let mut enuc = 0.0;

    for a in 0..coords.len() {
        for b in 0..a {
            let dx = coords[a][0] - coords[b][0];
            let dy = coords[a][1] - coords[b][1];
            let dz = coords[a][2] - coords[b][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            enuc += charges[a] * charges[b] / r;
        }
    }

    enuc
}

/// Build lightweight AO labels preserving atom-index parsing semantics.
/// # Arguments:
/// - `cint`: Integral environment.
/// # Returns:
/// - `Vec<String>`: AO labels whose first field is the atom index.
fn ao_labels(cint: &CInt) -> Vec<String> {
    let mut labels = Vec::with_capacity(cint.nao());

    for (a, slice) in cint.aoslice_by_atom().iter().enumerate() {
        for i in slice[2]..slice[3] {
            labels.push(format!("{a} AO{i}"));
        }
    }

    labels
}

/// Construct a spin-summed isolated-atom core-Hamiltonian density.
/// Degenerate orbitals are fractionally occupied to preserve spherical symmetry.
/// # Arguments:
/// - `symbol`: Atomic element symbol.
/// - `basis`: Basis-set name.
/// - `nelec`: Number of electrons in the neutral atom.
/// # Returns:
/// - `Array2<f64>`: Atomic AO density matrix.
fn atomic_core_density(
    symbol: &str,
    basis: &str,
    nelec: usize,
) -> Array2<f64> {
    let mol = CIntMol::from_toml(&format!(
        r#"
        atom = "{symbol} 0 0 0"
        basis = "{basis}"
        unit = "bohr"
        cart = false
        "#
    ));
    let cint = &mol.cint;

    let s = integral2(cint, "int1e_ovlp");
    let t = integral2(cint, "int1e_kin");
    let v = integral2(cint, "int1e_nuc");
    let h = &t + &v;
    let x = loewdin_x(&s, false, 1e-12);
    let (e, c) = general_evp_x(&h, &x);

    let mut occ = Array1::zeros(e.len());
    let mut remaining = nelec as f64;
    let mut i = 0;

    while i < e.len() && remaining > 0.0 {
        let mut j = i + 1;

        while j < e.len() && (e[j] - e[i]).abs() < 1e-8 {
            j += 1;
        }

        let degeneracy = j - i;
        let fill = remaining.min(2.0 * degeneracy as f64);
        let occupation = fill / degeneracy as f64;

        occ.slice_mut(s![i..j]).fill(occupation);
        remaining -= fill;
        i = j;
    }

    let mut weighted = c.clone();

    for (mut column, &occupation) in weighted.axis_iter_mut(Axis(1)).zip(occ.iter()) {
        column *= occupation;
    }

    weighted.dot(&c.t())
}

/// Construct the spin-summed superposition-of-atomic-densities guess.
/// # Arguments:
/// - `mol`: Parsed molecular libcint object.
/// - `basis`: Molecular basis-set name.
/// # Returns:
/// - `Array2<f64>`: Spin-summed SAD density matrix.
fn sad_density(
    mol: &CIntMol,
    basis: &str,
) -> Array2<f64> {
    let cint = &mol.cint;
    let aoslices = cint.aoslice_by_atom();
    let mut cache: HashMap<String, Array2<f64>> = HashMap::new();
    let mut dm = Array2::zeros((cint.nao(), cint.nao()));

    for (a, atom) in mol.atoms.iter().enumerate() {
        if atom.is_ghost {
            continue;
        }

        let atomic_dm = cache
            .entry(atom.symbol.clone())
            .or_insert_with(|| atomic_core_density(&atom.symbol, basis, atom.charge as usize));

        let ao_start = aoslices[a][2];
        let ao_end = aoslices[a][3];

        dm.slice_mut(s![ao_start..ao_end, ao_start..ao_end])
            .assign(atomic_dm);
    }

    dm
}

/// Generate AO integrals and a SAD initial density directly in memory.
/// # Arguments:
/// - `atoms`: Atom specifications in the input geometry.
/// - `basis`: Basis-set name.
/// - `unit`: Coordinate unit string.
/// # Returns:
/// - `AoData`: AO integrals, metadata, and atomic SAD density.
pub fn generate_ao_data(
    atoms: &[String],
    basis: &str,
    unit: &str,
) -> AoData {
    let unit = match unit {
        "Ang" | "ANG" | "angstrom" | "Angstrom" => "angstrom",
        "Bohr" | "BOHR" | "au" | "AU" | "a.u." | "A.U." => "bohr",
        _ => unit,
    };
    let atom = atoms.join("; ");
    let mol = CIntMol::from_toml(&format!(
        r#"
        atom = "{atom}"
        basis = "{basis}"
        unit = "{unit}"
        cart = false
        "#
    ));
    let cint = &mol.cint;

    let s = integral2(cint, "int1e_ovlp");
    let t = integral2(cint, "int1e_kin");
    let v = integral2(cint, "int1e_nuc");
    let h = &t + &v;
    let x = loewdin_x(&s, false, 1e-12);
    let dm = sad_density(&mol, basis);

    let eri_coul = integral4(cint, "int2e");
    let eri_asym = antisymmetrise(&eri_coul);

    let nelec = cint.atom_charges().iter().sum::<f64>().round() as i64;

    AoData {
        s,
        x,
        h,
        dm,
        eri_coul,
        eri_asym,
        enuc: nuclear_repulsion(cint),
        n: cint.nao(),
        nelec: Array1::from_vec(vec![nelec / 2, nelec / 2]),
        labels: ao_labels(cint),
    }
}
