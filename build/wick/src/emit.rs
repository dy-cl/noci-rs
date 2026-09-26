// emit.rs
//! Runtime term tables of the spin-free GNOCC residuals and FOIS metric.
//!
//! Every table is derived in the antisymmetrised spin-orbital basis, spin adapted, reduced
//! modulo exact spin relations and encoded over class-local index ids for `src/nocc`.

// Standard library imports.
use std::collections::BTreeMap;

// External crate imports.
use num_rational::Ratio;
use rayon::prelude::*;

// Crate-root imports.
use crate::canon::Key;
use crate::schema::{
    GeneratedTerm, OverlapBlockTerms, OverlapTermSet, ResidualClassTerms, TensorFactor,
};
use crate::specs::{self, Space};
use crate::spin::{self, DELTA, Table};
use crate::{reduce, so};

/// Return the spin-orbital class that produces one spin-free class.
/// # Arguments:
/// - `class`: Spin-free excitation class name.
/// # Returns:
/// - `&str`: Spin-orbital class name.
fn spin_orbital_class(class: &str) -> &str {
    if class == "CAToVA" { "CAToAV" } else { class }
}

/// Return the runtime id of one orbital space.
/// # Arguments:
/// - `s`: Orbital space.
/// # Returns:
/// - `u8`: Runtime space id.
fn space_id(s: Space) -> u8 {
    match s {
        Space::Core => 0,
        Space::Active => 1,
        Space::Virtual => 2,
    }
}

/// Return the runtime orbital-space ids by name.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Space-name map.
fn space_kind_table() -> BTreeMap<String, u8> {
    [("core", 0), ("active", 1), ("virtual", 2)]
        .into_iter()
        .map(|(n, k)| (n.to_string(), k))
        .collect()
}

/// Return the runtime tensor ids by name.
/// # Arguments:
/// - None.
/// # Returns:
/// - `BTreeMap<String, u8>`: Tensor-name map.
fn tensor_kind_table() -> BTreeMap<String, u8> {
    [
        ("Gamma1", spin::GAMMA),
        ("Theta", spin::THETA),
        ("f", spin::FOCK),
        ("g", spin::ERI),
        ("Lambda2", spin::LAMBDA2),
        ("Lambda3", spin::LAMBDA3),
        ("Lambda4", spin::LAMBDA4),
        ("t1", spin::T1),
        ("t2", spin::T2),
    ]
    .into_iter()
    .map(|(n, k)| (n.to_string(), k))
    .collect()
}

/// Generate the spin-free residual of one excitation class at one order in `T`.
/// # Arguments:
/// - `order`: Order in `T`.
/// - `class`: Spin-free excitation class name.
/// # Returns:
/// - `ResidualClassTerms`: Runtime residual terms.
/// # Panics
/// - Panics if `class` is not a known excitation class.
pub fn residual_class_terms(
    order: u8,
    class: &str,
) -> ResidualClassTerms {
    let bra = so::ops::projector_for_class(spin_orbital_class(class))
        .unwrap_or_else(|| panic!("unknown excitation class {class}"));
    let expr = so::wick::residual_expression(&bra, order as usize);

    let mut residual = spin::adapt_residual(spin_orbital_class(class), &expr)
        .into_iter()
        .find(|r| r.name == class)
        .unwrap_or_else(|| panic!("spin adaptation produced no {class} residual"));
    reduce::reduce_by_cumulant_relations(&mut residual);

    let names = specs::EXCS
        .iter()
        .find(|x| x.name == class)
        .map(|x| x.f)
        .unwrap_or_else(|| panic!("unknown excitation class {class}"));

    encode_table(&residual, names)
}

/// Derive one spin-free FOIS metric block before reduction,
/// `S_{\mu\nu} = \langle\Phi|\hat\tau_\mu^\dagger\hat\tau_\nu|\Phi\rangle`.
/// Free indices are those of the left class followed by those of the right class.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `Table`: Spin-free metric block.
/// # Panics
/// - Panics if `name` is not a known metric block.
pub(crate) fn metric_table(name: &str) -> Table {
    let b = specs::metric_block_spec(name);
    let bra = so::ops::projector_for_class(spin_orbital_class(b.left))
        .unwrap_or_else(|| panic!("unknown excitation class {}", b.left));
    let ket = so::ops::excitation_for_class(spin_orbital_class(b.right))
        .unwrap_or_else(|| panic!("unknown excitation class {}", b.right));

    spin::adapt_metric_block(
        b.name,
        b.left,
        b.right,
        &so::wick::metric_expression(&bra, &ket),
    )
}

/// Generate one spin-free FOIS metric block.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime metric terms.
/// # Panics
/// - Panics if `name` is not a known metric block.
pub fn overlap_block_terms(name: &str) -> OverlapBlockTerms {
    let mut block = metric_table(name);
    reduce::reduce_by_cumulant_relations(&mut block);

    encode_block(name, &block)
}

/// Generate every spin-free FOIS metric block.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Complete metric term table.
pub fn overlap_terms() -> OverlapTermSet {
    OverlapTermSet {
        version: 1,
        space_kinds: space_kind_table(),
        tensor_kinds: tensor_kind_table(),
        blocks: specs::BLOCKS
            .par_iter()
            .map(|b| (b.name.to_string(), overlap_block_terms(b.name)))
            .collect(),
    }
}

/// Generate one spin-free zeroth-order coupling block
/// `\langle\Phi|\hat\tau_\mu^\dagger\hat H_0\hat\tau_\nu|\Phi\rangle_c` of the Dyall Hamiltonian.
/// The blocks follow the metric blocks, since `\hat H_0` conserves the number of electrons in
/// every orbital space.
/// # Arguments:
/// - `name`: Metric block name.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime coupling terms.
/// # Panics
/// - Panics if `name` is not a known metric block.
pub fn dyall_block_terms(name: &str) -> OverlapBlockTerms {
    let b = specs::metric_block_spec(name);
    let bra = so::ops::projector_for_class(spin_orbital_class(b.left))
        .unwrap_or_else(|| panic!("unknown excitation class {}", b.left));
    let ket = so::ops::excitation_for_class(spin_orbital_class(b.right))
        .unwrap_or_else(|| panic!("unknown excitation class {}", b.right));

    let mut block = spin::adapt_metric_block(
        b.name,
        b.left,
        b.right,
        &so::wick::dyall_coupling_expression(&bra, &ket),
    );
    reduce::reduce_by_cumulant_relations(&mut block);

    encode_block(name, &block)
}

/// Generate every spin-free zeroth-order Dyall coupling block.
/// # Arguments:
/// - None.
/// # Returns:
/// - `OverlapTermSet`: Complete coupling term table.
pub fn dyall_terms() -> OverlapTermSet {
    OverlapTermSet {
        version: 1,
        space_kinds: space_kind_table(),
        tensor_kinds: tensor_kind_table(),
        blocks: specs::BLOCKS
            .par_iter()
            .map(|b| (b.name.to_string(), dyall_block_terms(b.name)))
            .collect(),
    }
}

/// Generate the spin-free correlation energy at one order in `T`,
/// `E_1 = \langle\Phi|\hat H\hat T|\Phi\rangle_c` or
/// `E_2 = \tfrac12\langle\Phi|\hat H\{\hat T\hat T\}|\Phi\rangle_c`.
/// # Arguments:
/// - `order`: Order in `T`, `1` or `2`.
/// # Returns:
/// - `ResidualClassTerms`: Runtime terms with no free indices.
pub fn energy_terms(order: u8) -> ResidualClassTerms {
    let expr = so::wick::energy_expression(order as usize);
    let mut energy = spin::adapt_scalar("energy", &expr);
    reduce::reduce_by_cumulant_relations(&mut energy);

    encode_table(&energy, &[])
}

/// Encode one spin-free block with left and right free indices as runtime terms.
/// # Arguments:
/// - `name`: Metric block name.
/// - `block`: Spin-free block over the left then right free indices.
/// # Returns:
/// - `OverlapBlockTerms`: Runtime block terms.
fn encode_block(
    name: &str,
    block: &Table,
) -> OverlapBlockTerms {
    let b = specs::metric_block_spec(name);
    let names = [b.lf, b.rf].concat();
    let (nl, nr) = (b.lf.len() as u16, b.rf.len() as u16);
    let t = encode_table(block, &names);

    OverlapBlockTerms {
        left: b.left.to_string(),
        right: b.right.to_string(),
        indices: t.indices,
        left_free: (0..nl).collect(),
        right_free: (nl..nl + nr).collect(),
        terms: t.terms,
    }
}

/// Encode one spin-free table as runtime terms.
/// Free indices keep ids `0..n` in layout order. Dummy indices of each term are mapped to
/// shared indices by `(space, rank within that space)`, so terms reuse loop slots.
/// # Arguments:
/// - `res`: Spin-free table.
/// - `names`: Free-index names in layout order.
/// # Returns:
/// - `ResidualClassTerms`: Runtime terms in deterministic key order.
fn encode_table(
    res: &Table,
    names: &[&str],
) -> ResidualClassTerms {
    let nfree = res.free.len();
    let mut indices = names
        .iter()
        .zip(&res.free)
        .map(|(&n, &s)| (n.to_string(), space_id(s)))
        .collect::<Vec<_>>();
    let mut slots = BTreeMap::<(u8, usize), u16>::new();

    // Sort keys so the output order is deterministic.
    let mut keys = res.terms.iter().collect::<Vec<_>>();
    keys.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let terms = keys
        .into_iter()
        .map(|(key, &c)| encode_term(key, c, nfree, &mut slots, &mut indices))
        .collect();

    ResidualClassTerms {
        indices,
        free: (0..nfree as u16).collect(),
        terms,
    }
}

/// Encode one canonical spin-free term; delta factors become runtime deltas.
/// # Arguments:
/// - `key`: Canonical term.
/// - `c`: Term coefficient.
/// - `nfree`: Number of free indices.
/// - `slots`: Shared dummy slots keyed by `(space, rank)`.
/// - `indices`: Class index table, extended with new dummy slots.
/// # Returns:
/// - `GeneratedTerm`: Runtime term.
fn encode_term(
    key: &Key,
    c: Ratio<i64>,
    nfree: usize,
    slots: &mut BTreeMap<(u8, usize), u16>,
    indices: &mut Vec<(String, u8)>,
) -> GeneratedTerm {
    // Map each dummy to the shared slot of its space and rank.
    let mut rank = [0usize; 3];
    let mut map = (0..nfree as u16).collect::<Vec<_>>();
    for &s in &key.dummies {
        let r = rank[s as usize];
        rank[s as usize] += 1;
        let id = *slots.entry((s, r)).or_insert_with(|| {
            indices.push((format!("d{s}{r}"), s));
            (indices.len() - 1) as u16
        });
        map.push(id);
    }

    let (deltas, tensors): (Vec<_>, Vec<_>) = key.factors.iter().partition(|f| f.kind == DELTA);

    GeneratedTerm(
        [*c.numer(), *c.denom()],
        map[nfree..].to_vec(),
        deltas
            .iter()
            .map(|f| [map[f.upper[0] as usize], map[f.lower[0] as usize]])
            .collect(),
        tensors
            .iter()
            .map(|f| {
                TensorFactor(
                    f.kind,
                    f.upper.iter().map(|&x| map[x as usize]).collect(),
                    f.lower.iter().map(|&x| map[x as usize]).collect(),
                )
            })
            .collect(),
    )
}
