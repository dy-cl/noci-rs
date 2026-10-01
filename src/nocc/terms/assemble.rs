// nocc/terms/assemble.rs
//! Assembly of generated term tables into scalars, vectors and matrices over the raw
//! excitation basis.
//!
//! Every assembled quantity follows one pipeline: group the raw excitations by class, plan
//! every table, build one set of dense factor blocks shared by all tables of the call, evaluate
//! each table as a dense block over its free indices, and gather the listed excitations.

// Standard library imports.
use std::collections::BTreeMap;
use std::sync::Arc;

// External crate imports.
use ndarray::{Array1, Array2};

// Crate-root imports.
use crate::NOCIScalar;
use crate::nocc::space::{Excitation, ExcitationClass, Spaces, excitation_class};

// Parent/sibling imports.
use super::evaluate::evaluate_dense_table;
use super::factors::FactorBlocks;
use super::loader::{BlockTables, ClassTables};
use super::plan::{TablePlan, TermEvaluator, TermTable};
use super::schema::ResidualClassTerms;
use super::tensors::{Tensors, excitation_indices};

/// Plan lookup of the evaluator, `TermEvaluator::table_plan` for the truncated working
/// equations or `TermEvaluator::exact_table_plan` for exact reference properties.
pub(in crate::nocc) type PlanChoice = fn(&TermEvaluator, TermTable<'_>) -> Arc<TablePlan>;

/// Return the position of every orbital within its own orbital space.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// # Returns:
/// - `Vec<usize>`: Position of each MO in the core, active or virtual list.
fn orbital_positions(spaces: &Spaces) -> Vec<usize> {
    let mut positions = vec![0; spaces.nmo];
    for list in [&spaces.core, &spaces.active, &spaces.virtuals] {
        for (i, &p) in list.iter().enumerate() {
            positions[p] = i;
        }
    }
    positions
}

/// Group the raw excitations by excitation class.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// # Returns:
/// - `BTreeMap<ExcitationClass, Vec<usize>>`: Positions in the list of every class's
///   excitations.
fn class_members(
    spaces: &Spaces,
    excitations: &[Excitation],
) -> BTreeMap<ExcitationClass, Vec<usize>> {
    let mut members = BTreeMap::<ExcitationClass, Vec<usize>>::new();
    for (mu, &ex) in excitations.iter().enumerate() {
        members
            .entry(excitation_class(spaces, ex))
            .or_default()
            .push(mu);
    }
    members
}

/// Evaluate the sum of several class-keyed table orders at every raw excitation.
/// Each excitation class is evaluated as one dense block over its free-index spaces, summed
/// over the orders in the given sequence, and its listed elements are then gathered.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `plan`: Plan lookup, truncated or exact.
/// - `orders`: Class-keyed tables of the orders to sum.
/// - `tensors`: Runtime tensors, including the amplitudes when any order needs them.
/// # Returns:
/// - `Array1<T>`: Summed quantity in the raw excitation basis, in the amplitude scalar type.
/// # Panics
/// - Panics if `orders` is empty or an order has no terms for an excitation class in the list.
pub(in crate::nocc) fn assemble_vector<T: NOCIScalar>(
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    plan: PlanChoice,
    orders: &[&ClassTables],
    tensors: &Tensors<'_>,
) -> Array1<T> {
    // Group the raw excitations by class.
    let classes = class_members(spaces, excitations);

    // Dense factor blocks shared by every table of every class.
    let plans = classes
        .keys()
        .flat_map(|class| orders.iter().map(move |order| order[class]))
        .map(|c| plan(evaluator, (&c.terms, &c.indices)))
        .collect::<Vec<_>>();
    let blocks = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        tensors,
    );

    let positions = orbital_positions(spaces);
    let mut out = Array1::<T>::zeros(excitations.len());

    for (class, members) in &classes {
        // `X = \sum_n X_n` over the requested orders, as one dense block.
        let mut block: Option<(Vec<T>, Vec<usize>)> = None;
        for order in orders {
            let terms = order[class];
            let table = (terms.terms.as_slice(), terms.indices.as_slice());
            let table_plan = plan(evaluator, table);
            let part = evaluate_dense_table(table, &terms.free, &table_plan, &blocks);
            let data = part.data.into_elements::<T>();
            match &mut block {
                Some((b, _)) => {
                    for (x, y) in b.iter_mut().zip(data) {
                        *x += y;
                    }
                }
                None => block = Some((data, part.dims)),
            }
        }
        let (data, dims) = block.expect("at least one table order");

        // Gather each listed excitation from its free-index tuple.
        for &mu in members {
            let (values, n) = excitation_indices(excitations[mu]);
            let flat = values[..n]
                .iter()
                .zip(&dims)
                .fold(0, |acc, (&p, &d)| acc * d + positions[p]);
            out[mu] = data[flat];
        }
    }

    out
}

/// Assemble a symmetric matrix over the raw excitations from its generated class-pair blocks.
/// Each block whose classes both occur is evaluated once as a dense tensor over its left then
/// right free indices, and every listed pair of excitations is gathered from it. Class pairs
/// without a block couple to zero, and each block also fills its transpose.
/// # Arguments:
/// - `spaces`: Core, active, and virtual orbital-space maps.
/// - `excitations`: Raw spin-free excitation list.
/// - `evaluator`: Term-table evaluator.
/// - `plan`: Plan lookup, truncated or exact.
/// - `set`: Generated class-pair blocks, such as the metric or the Dyall coupling.
/// - `tensors`: Runtime tensors.
/// # Returns:
/// - `Array2<f64>`: Matrix over the raw excitation list.
pub(in crate::nocc) fn assemble_matrix(
    spaces: &Spaces,
    excitations: &[Excitation],
    evaluator: &TermEvaluator,
    plan: PlanChoice,
    set: &BlockTables,
    tensors: &Tensors<'_>,
) -> Array2<f64> {
    // Group the raw excitations by class.
    let members = class_members(spaces, excitations);

    // Blocks whose classes both occur, with dense factor blocks shared by all of them.
    let blocks = set
        .iter()
        .filter(|(left, right, _)| members.contains_key(left) && members.contains_key(right))
        .collect::<Vec<_>>();
    let plans = blocks
        .iter()
        .map(|(_, _, b)| plan(evaluator, (&b.terms, &b.indices)))
        .collect::<Vec<_>>();
    let factors = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        tensors,
    );

    let positions = orbital_positions(spaces);
    let n = excitations.len();
    let mut out = Array2::<f64>::zeros((n, n));

    for (&&(left, right, block), table_plan) in blocks.iter().zip(&plans) {
        let free = [block.left_free.as_slice(), block.right_free.as_slice()].concat();
        let dense =
            evaluate_dense_table((&block.terms, &block.indices), &free, table_plan, &factors);
        let data = dense.data.into_elements::<f64>();

        // Gather every pair from the left then right free-index tuple.
        for &mu in &members[&left] {
            let (left, nl) = excitation_indices(excitations[mu]);
            for &nu in &members[&right] {
                let (right, nr) = excitation_indices(excitations[nu]);
                let flat = left[..nl]
                    .iter()
                    .chain(&right[..nr])
                    .zip(&dense.dims)
                    .fold(0, |acc, (&p, &d)| acc * d + positions[p]);
                out[(mu, nu)] = data[flat];
                out[(nu, mu)] = data[flat];
            }
        }
    }

    out
}

/// Evaluate the sum of several tables without free indices.
/// # Arguments:
/// - `evaluator`: Term-table evaluator.
/// - `plan`: Plan lookup, truncated or exact.
/// - `tables`: Generated tables to sum, in order.
/// - `tensors`: Runtime tensors.
/// # Returns:
/// - `T`: Sum of the table values, in the amplitude scalar type.
pub(in crate::nocc) fn assemble_scalar<T: NOCIScalar>(
    evaluator: &TermEvaluator,
    plan: PlanChoice,
    tables: &[&ResidualClassTerms],
    tensors: &Tensors<'_>,
) -> T {
    // Dense factor blocks shared by every table.
    let tables = tables
        .iter()
        .map(|t| (t.terms.as_slice(), t.indices.as_slice()))
        .collect::<Vec<_>>();
    let plans = tables
        .iter()
        .map(|&t| plan(evaluator, t))
        .collect::<Vec<_>>();
    let blocks = FactorBlocks::build_factor_blocks(
        &plans.iter().map(|p| p.as_ref()).collect::<Vec<_>>(),
        tensors,
    );

    tables
        .iter()
        .zip(&plans)
        .map(|(&t, table_plan)| {
            evaluate_dense_table(t, &[], table_plan, &blocks)
                .data
                .into_elements::<T>()[0]
        })
        .fold(<T as From<f64>>::from(0.0), |acc, x| acc + x)
}
