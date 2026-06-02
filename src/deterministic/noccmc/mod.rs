// deterministic/noccmc/mod.rs

mod overlap;
mod space;

use mpi::topology::Communicator;
use ndarray::Array1;

use crate::PostSCFData;
use crate::input::Input;
use crate::maths::general_evp;
use crate::noci::{
    Cumulants, NOCIData, RDM1, RDM2, RDM3, RDM4, build_noci_hs, build_wicks_shared, cumulants,
    rdm1, rdm2, rdm3, rdm4,
};
use crate::nonorthogonalwicks::WickScratchSpin;
use crate::orbitals::{
    NOCINaturalOrbitals, print_noci_natural_orbitals, transform_ao_data, transform_noci_basis,
};

/// Run NOCCMC setup work.
/// # Arguments:
/// - `post`: Data shared by post-SCF methods.
/// - `input`: User input specifications.
/// - `c0`: Reference NOCI coefficient vector.
/// - `no`: NOCI natural orbital basis.
/// - `world`: MPI communicator object.
/// # Returns:
/// - `()`: Builds and checks spin-free RDMs, cumulants, and the first weighted FOIS metric block.
pub(crate) fn run_noccmc(
    post: &PostSCFData<'_, f64>,
    input: &Input,
    c0: &[f64],
    no: &NOCINaturalOrbitals,
    world: &impl Communicator,
) {
    let coeffs = Array1::from_vec(c0.to_vec());

    if world.rank() == 0 {
        print_noci_natural_orbitals("NOCI natural orbitals", no);
    }

    let nobasis = transform_noci_basis(post.noci_reference_basis, &no.c, &post.ao.s);
    let noao = transform_ao_data(post.ao, &no.c);

    let nowicks = if input.wicks.enabled {
        Some(build_wicks_shared(world, &noao, &nobasis, post.tol, input))
    } else {
        None
    };

    let nowicksview = nowicks.as_ref().map(|w| w.view());
    let nodata = NOCIData::new(&noao, &nobasis, input, post.tol, nowicksview);

    if world.rank() == 0 {
        println!("{}", "=".repeat(100));
        println!("Running NOCCMC spin-free RDM check in NOCI natural orbital basis....");
    }

    let mut scratch1 = WickScratchSpin::new();
    let scratch1 = if input.wicks.enabled {
        Some(&mut scratch1)
    } else {
        None
    };
    let (_, gamma1) = rdm1(&nodata, &coeffs, &coeffs, scratch1);

    let mut scratch2 = WickScratchSpin::new();
    let scratch2 = if input.wicks.enabled {
        Some(&mut scratch2)
    } else {
        None
    };
    let (_, gamma2) = rdm2(&nodata, &coeffs, &coeffs, scratch2);

    let mut scratch3 = WickScratchSpin::new();
    let scratch3 = if input.wicks.enabled {
        Some(&mut scratch3)
    } else {
        None
    };
    let (_, gamma3) = rdm3(&nodata, &coeffs, &coeffs, &no.active, scratch3);

    let mut scratch4 = WickScratchSpin::new();
    let scratch4 = if input.wicks.enabled {
        Some(&mut scratch4)
    } else {
        None
    };
    let (_, gamma4) = rdm4(&nodata, &coeffs, &coeffs, &no.active, scratch4);

    if world.rank() == 0 {
        let lambdas = cumulants(&gamma1, &gamma2, &gamma3, &gamma4, &no.active);

        let mut scheck = no.c.t().dot(&post.ao.s).dot(&no.c);
        for i in 0..scheck.nrows() {
            scheck[(i, i)] -= 1.0;
        }

        let serr = scheck.iter().map(|x| x.abs()).fold(0.0, f64::max);

        let (h, s, _) = build_noci_hs(&nodata, nodata.basis, nodata.basis, true);
        let e_coeff = coeffs.dot(&h.dot(&coeffs)) / coeffs.dot(&s.dot(&coeffs));
        let (evals, _) = general_evp(&h, &s, true, post.tol);

        let mut e1 = 0.0;
        for a in 0..gamma1.n {
            for b in 0..gamma1.n {
                let i = b * gamma1.n + a;
                e1 += noao.h[(a, b)] * gamma1.data[i];
            }
        }

        let mut e2 = 0.0;
        for a in 0..gamma2.n {
            for b in 0..gamma2.n {
                for c in 0..gamma2.n {
                    for d in 0..gamma2.n {
                        let i = (((b * gamma2.n + c) * gamma2.n + a) * gamma2.n) + d;
                        e2 += noao.eri_coul[(a, b, c, d)] * gamma2.data[i];
                    }
                }
            }
        }

        let erdm = noao.enuc + e1 + 0.5 * e2;

        print_misc_diagnostics(serr, e_coeff, evals[0], erdm);
        print_rdm_diagnostics(&gamma1, &gamma2, &gamma3, &gamma4, &no.active);
        print_cumulant_diagnostics(&gamma1, &gamma2, &gamma3, &gamma4, &lambdas, &no.active);

        let spaces = space::build_spaces(gamma1.n, &no.active, &gamma1, 1.0e-6, 1.0e-6);
        let excitations = space::build_excitations(&spaces);

        space::print_space_diagnostics(&spaces, &excitations);
        space::print_fois_metric_diagnostics(
            &noao,
            &gamma1,
            &lambdas,
            &spaces,
            &excitations,
            post.tol,
        );
    }
}

/// Print miscellaneous NOCCMC setup diagnostics.
/// # Arguments:
/// - `serr`: NOCI natural orbital orthonormality error.
/// - `e_coeff`: NOCI energy evaluated from the supplied coefficient vector.
/// - `e_gep`: Lowest NOCI generalized eigenvalue.
/// - `e_rdm`: NOCI energy reconstructed from the spin-free one- and two-body RDMs.
/// # Returns:
/// - `()`: Prints miscellaneous diagnostics.
fn print_misc_diagnostics(
    serr: f64,
    e_coeff: f64,
    e_gep: f64,
    e_rdm: f64,
) {
    println!("{}", "=".repeat(100));
    println!("NOCI NOCCMC miscellaneous diagnostics");
    println!("NOCI natural orbital orthonormality error: {:.6e}", serr);
    println!("NOCI energy in NO basis: {:.12}", e_coeff);
    println!("Lowest NOCI GEVP energy in NO basis: {:.12}", e_gep);
    println!("NOCI energy from RDMs in NO basis: {:.12}", e_rdm);
}

/// Print spin-free RDM consistency diagnostics.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `gamma2`: Full-space spin-free two-body RDM.
/// - `gamma3`: Active-space spin-free three-body RDM.
/// - `gamma4`: Active-space spin-free four-body RDM.
/// - `active`: Active orbital indices in the full NO basis.
/// # Returns:
/// - `()`: Prints RDM diagnostics.
fn print_rdm_diagnostics(
    gamma1: &RDM1<f64>,
    gamma2: &RDM2<f64>,
    gamma3: &RDM3<f64>,
    gamma4: &RDM4<f64>,
    active: &[usize],
) {
    let n = active.len();

    let mut nelec = 0.0;
    for p in 0..gamma1.n {
        nelec += gamma1.data[p * gamma1.n + p];
    }

    let mut nact = 0.0;
    for &p in active.iter() {
        nact += gamma1.data[p * gamma1.n + p];
    }

    let mut tr2 = 0.0;
    for p in 0..gamma2.n {
        for q in 0..gamma2.n {
            let i = (((p * gamma2.n + q) * gamma2.n + p) * gamma2.n) + q;
            tr2 += gamma2.data[i];
        }
    }

    let mut tr3a = 0.0;
    for p in 0..gamma3.n {
        for q in 0..gamma3.n {
            for r in 0..gamma3.n {
                let i = (((((p * gamma3.n + q) * gamma3.n + r) * gamma3.n + p) * gamma3.n + q)
                    * gamma3.n)
                    + r;
                tr3a += gamma3.data[i];
            }
        }
    }

    let mut tr4a = 0.0;
    for p in 0..gamma4.n {
        for q in 0..gamma4.n {
            for r in 0..gamma4.n {
                for s in 0..gamma4.n {
                    let i = (((((((p * gamma4.n + q) * gamma4.n + r) * gamma4.n + s) * gamma4.n
                        + p)
                        * gamma4.n
                        + q)
                        * gamma4.n
                        + r)
                        * gamma4.n)
                        + s;
                    tr4a += gamma4.data[i];
                }
            }
        }
    }

    let mut g2_contract_err: f64 = 0.0;
    for p in 0..gamma1.n {
        for r in 0..gamma1.n {
            let mut lhs = 0.0;

            for q in 0..gamma1.n {
                let i = (((p * gamma2.n + q) * gamma2.n + r) * gamma2.n) + q;
                lhs += gamma2.data[i];
            }

            let rhs = (nelec - 1.0) * gamma1.data[p * gamma1.n + r];
            g2_contract_err = g2_contract_err.max((lhs - rhs).abs());
        }
    }

    let mut g3_active_contract_err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let mut lhs = 0.0;

                    for t in 0..n {
                        let i = (((((p * gamma3.n + q) * gamma3.n + t) * gamma3.n + r) * gamma3.n
                            + s)
                            * gamma3.n)
                            + t;
                        lhs += gamma3.data[i];
                    }

                    let pp = active[p];
                    let qq = active[q];
                    let rr = active[r];
                    let ss = active[s];
                    let g2i = (((pp * gamma2.n + qq) * gamma2.n + rr) * gamma2.n) + ss;
                    let rhs = (nact - 2.0) * gamma2.data[g2i];

                    g3_active_contract_err = g3_active_contract_err.max((lhs - rhs).abs());
                }
            }
        }
    }

    let mut g4_active_contract_err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    for t in 0..n {
                        for u in 0..n {
                            let mut lhs = 0.0;

                            for v in 0..n {
                                let i = (((((((p * gamma4.n + q) * gamma4.n + r) * gamma4.n
                                    + v)
                                    * gamma4.n
                                    + s)
                                    * gamma4.n
                                    + t)
                                    * gamma4.n
                                    + u)
                                    * gamma4.n)
                                    + v;
                                lhs += gamma4.data[i];
                            }

                            let g3i = (((((p * gamma3.n + q) * gamma3.n + r) * gamma3.n + s)
                                * gamma3.n
                                + t)
                                * gamma3.n)
                                + u;
                            let rhs = (nact - 3.0) * gamma3.data[g3i];

                            g4_active_contract_err = g4_active_contract_err.max((lhs - rhs).abs());
                        }
                    }
                }
            }
        }
    }

    println!("{}", "=".repeat(100));
    println!("NOCI spin-free RDM diagnostics");
    println!("Trace Gamma1: {:.10}", nelec);
    println!("Trace active Gamma1: {:.10}", nact);
    println!(
        "Trace Gamma2: {:.10} expected full N(N-1): {:.10}",
        tr2,
        nelec * (nelec - 1.0)
    );
    println!(
        "Trace active Gamma3: {:.10} fixed-active estimate: {:.10}",
        tr3a,
        nact * (nact - 1.0) * (nact - 2.0)
    );
    println!(
        "Trace active Gamma4: {:.10} fixed-active estimate: {:.10}",
        tr4a,
        nact * (nact - 1.0) * (nact - 2.0) * (nact - 3.0)
    );
    println!("Max full Gamma2 contraction error: {:.6e}", g2_contract_err);
    println!(
        "Max active Gamma3 -> Gamma2 contraction error: {:.6e}",
        g3_active_contract_err
    );
    println!(
        "Max active Gamma4 -> Gamma3 contraction error: {:.6e}",
        g4_active_contract_err
    );
    println!(
        "Max |Gamma1|: {:.6e}",
        gamma1.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    );
    println!(
        "Max |Gamma2|: {:.6e}",
        gamma2.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    );
    println!(
        "Max active |Gamma3|: {:.6e}",
        gamma3.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    );
    println!(
        "Max active |Gamma4|: {:.6e}",
        gamma4.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    );
}

/// Print spin-free cumulant consistency diagnostics.
/// # Arguments:
/// - `gamma1`: Full-space spin-free one-body RDM.
/// - `gamma2`: Full-space spin-free two-body RDM.
/// - `gamma3`: Active-space spin-free three-body RDM.
/// - `gamma4`: Active-space spin-free four-body RDM.
/// - `lambda`: Active-space spin-free cumulants.
/// - `active`: Active orbital indices in the full NO basis.
/// # Returns:
/// - `()`: Prints cumulant diagnostics.
fn print_cumulant_diagnostics(
    gamma1: &RDM1<f64>,
    gamma2: &RDM2<f64>,
    gamma3: &RDM3<f64>,
    gamma4: &RDM4<f64>,
    lambda: &Cumulants<f64>,
    active: &[usize],
) {
    let n = active.len();

    let half = 0.5;
    let quarter = 0.25;
    let eighth = 0.125;
    let third = 1.0 / 3.0;
    let sixth = 1.0 / 6.0;

    let l1 = |p: usize, q: usize| -> f64 { lambda.lambda1.get(&[p], &[q]) };
    let l2 =
        |p: usize, q: usize, r: usize, s: usize| -> f64 { lambda.lambda2.get(&[p, q], &[r, s]) };
    let l3 = |p: usize, q: usize, r: usize, s: usize, t: usize, u: usize| -> f64 {
        lambda.lambda3.get(&[p, q, r], &[s, t, u])
    };
    let l4 =
        |p: usize, q: usize, r: usize, w: usize, s: usize, t: usize, u: usize, v: usize| -> f64 {
            lambda.lambda4.get(&[p, q, r, w], &[s, t, u, v])
        };

    let mut l1err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            let refv = gamma1.data[active[p] * gamma1.n + active[q]];
            l1err = l1err.max((l1(p, q) - refv).abs());
        }
    }

    let mut l2err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let pp = active[p];
                    let qq = active[q];
                    let rr = active[r];
                    let ss = active[s];

                    let g2i = (((pp * gamma2.n + qq) * gamma2.n + rr) * gamma2.n) + ss;
                    let g1pr = gamma1.data[pp * gamma1.n + rr];
                    let g1qs = gamma1.data[qq * gamma1.n + ss];
                    let g1ps = gamma1.data[pp * gamma1.n + ss];
                    let g1qr = gamma1.data[qq * gamma1.n + rr];

                    let refv = gamma2.data[g2i] - g1pr * g1qs + half * g1ps * g1qr;
                    l2err = l2err.max((l2(p, q, r, s) - refv).abs());
                }
            }
        }
    }

    let mut l3err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    for t in 0..n {
                        for u in 0..n {
                            let pp = active[p];
                            let qq = active[q];
                            let rr = active[r];
                            let ss = active[s];
                            let tt = active[t];
                            let uu = active[u];

                            let g3i = (((((p * gamma3.n + q) * gamma3.n + r) * gamma3.n + s)
                                * gamma3.n
                                + t)
                                * gamma3.n)
                                + u;

                            let g1ps = gamma1.data[pp * gamma1.n + ss];
                            let g1pt = gamma1.data[pp * gamma1.n + tt];
                            let g1pu = gamma1.data[pp * gamma1.n + uu];

                            let g1qs = gamma1.data[qq * gamma1.n + ss];
                            let g1qt = gamma1.data[qq * gamma1.n + tt];
                            let g1qu = gamma1.data[qq * gamma1.n + uu];

                            let g1rs = gamma1.data[rr * gamma1.n + ss];
                            let g1rt = gamma1.data[rr * gamma1.n + tt];
                            let g1ru = gamma1.data[rr * gamma1.n + uu];

                            let disconnected = g1ps * g1qt * g1ru
                                - half * g1ps * g1qu * g1rt
                                - half * g1pt * g1qs * g1ru
                                - half * g1pu * g1qt * g1rs
                                + quarter * g1pt * g1qu * g1rs
                                + quarter * g1pu * g1qs * g1rt
                                + g1ps * l2(q, r, t, u)
                                - half * g1pt * l2(q, r, s, u)
                                - half * g1pu * l2(q, r, t, s)
                                + g1qt * l2(p, r, s, u)
                                - half * g1qs * l2(p, r, t, u)
                                - half * g1qu * l2(p, r, s, t)
                                + g1ru * l2(p, q, s, t)
                                - half * g1rs * l2(p, q, u, t)
                                - half * g1rt * l2(p, q, s, u);

                            let refv = gamma3.data[g3i] - disconnected;
                            l3err = l3err.max((l3(p, q, r, s, t, u) - refv).abs());
                        }
                    }
                }
            }
        }
    }

    let mut l4err: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for w in 0..n {
                    for s in 0..n {
                        for t in 0..n {
                            for u in 0..n {
                                for v in 0..n {
                                    let g4i = (((((((p * gamma4.n + q) * gamma4.n + r)
                                        * gamma4.n
                                        + w)
                                        * gamma4.n
                                        + s)
                                        * gamma4.n
                                        + t)
                                        * gamma4.n
                                        + u)
                                        * gamma4.n)
                                        + v;

                                    let disconnected = l1(p, s) * l1(q, t) * l1(r, u) * l1(w, v)
                                        - half * l1(p, s) * l1(q, t) * l1(r, v) * l1(w, u)
                                        - half * l1(p, s) * l1(q, u) * l1(r, t) * l1(w, v)
                                        + quarter * l1(p, s) * l1(q, u) * l1(r, v) * l1(w, t)
                                        + quarter * l1(p, s) * l1(q, v) * l1(r, t) * l1(w, u)
                                        - half * l1(p, s) * l1(q, v) * l1(r, u) * l1(w, t)
                                        - half * l1(p, t) * l1(q, s) * l1(r, u) * l1(w, v)
                                        + quarter * l1(p, t) * l1(q, s) * l1(r, v) * l1(w, u)
                                        + quarter * l1(p, t) * l1(q, u) * l1(r, s) * l1(w, v)
                                        - eighth * l1(p, t) * l1(q, u) * l1(r, v) * l1(w, s)
                                        - eighth * l1(p, t) * l1(q, v) * l1(r, s) * l1(w, u)
                                        + quarter * l1(p, t) * l1(q, v) * l1(r, u) * l1(w, s)
                                        + quarter * l1(p, u) * l1(q, s) * l1(r, t) * l1(w, v)
                                        - eighth * l1(p, u) * l1(q, s) * l1(r, v) * l1(w, t)
                                        - half * l1(p, u) * l1(q, t) * l1(r, s) * l1(w, v)
                                        + quarter * l1(p, u) * l1(q, t) * l1(r, v) * l1(w, s)
                                        + quarter * l1(p, u) * l1(q, v) * l1(r, s) * l1(w, t)
                                        - eighth * l1(p, u) * l1(q, v) * l1(r, t) * l1(w, s)
                                        - eighth * l1(p, v) * l1(q, s) * l1(r, t) * l1(w, u)
                                        + quarter * l1(p, v) * l1(q, s) * l1(r, u) * l1(w, t)
                                        + quarter * l1(p, v) * l1(q, t) * l1(r, s) * l1(w, u)
                                        - half * l1(p, v) * l1(q, t) * l1(r, u) * l1(w, s)
                                        - eighth * l1(p, v) * l1(q, u) * l1(r, s) * l1(w, t)
                                        + quarter * l1(p, v) * l1(q, u) * l1(r, t) * l1(w, s)
                                        + l1(p, s) * l1(q, t) * l2(r, w, u, v)
                                        - half * l1(p, s) * l1(q, u) * l2(r, w, t, v)
                                        - half * l1(p, s) * l1(q, v) * l2(r, w, u, t)
                                        - half * l1(p, s) * l1(r, t) * l2(q, w, u, v)
                                        + l1(p, s) * l1(r, u) * l2(q, w, t, v)
                                        - half * l1(p, s) * l1(r, v) * l2(q, w, t, u)
                                        - half * l1(p, s) * l1(w, t) * l2(q, r, v, u)
                                        - half * l1(p, s) * l1(w, u) * l2(q, r, t, v)
                                        + l1(p, s) * l1(w, v) * l2(q, r, t, u)
                                        - half * l1(p, t) * l1(q, s) * l2(r, w, u, v)
                                        + quarter * l1(p, t) * l1(q, u) * l2(r, w, s, v)
                                        + quarter * l1(p, t) * l1(q, v) * l2(r, w, u, s)
                                        + quarter * l1(p, t) * l1(r, s) * l2(q, w, u, v)
                                        - half * l1(p, t) * l1(r, u) * l2(q, w, s, v)
                                        + quarter * l1(p, t) * l1(r, v) * l2(q, w, s, u)
                                        + quarter * l1(p, t) * l1(w, s) * l2(q, r, v, u)
                                        + quarter * l1(p, t) * l1(w, u) * l2(q, r, s, v)
                                        - half * l1(p, t) * l1(w, v) * l2(q, r, s, u)
                                        + quarter * l1(p, u) * l1(q, s) * l2(r, w, t, v)
                                        - half * l1(p, u) * l1(q, t) * l2(r, w, s, v)
                                        + quarter * l1(p, u) * l1(q, v) * l2(r, w, s, t)
                                        - half * l1(p, u) * l1(r, s) * l2(q, w, t, v)
                                        + quarter * l1(p, u) * l1(r, t) * l2(q, w, s, v)
                                        + quarter * l1(p, u) * l1(r, v) * l2(q, w, t, s)
                                        + quarter * l1(p, u) * l1(w, s) * l2(q, r, t, v)
                                        + quarter * l1(p, u) * l1(w, t) * l2(q, r, v, s)
                                        - half * l1(p, u) * l1(w, v) * l2(q, r, t, s)
                                        + quarter * l1(p, v) * l1(q, s) * l2(r, w, u, t)
                                        - half * l1(p, v) * l1(q, t) * l2(r, w, u, s)
                                        + quarter * l1(p, v) * l1(q, u) * l2(r, w, t, s)
                                        + quarter * l1(p, v) * l1(r, s) * l2(q, w, t, u)
                                        + quarter * l1(p, v) * l1(r, t) * l2(q, w, u, s)
                                        - half * l1(p, v) * l1(r, u) * l2(q, w, t, s)
                                        - half * l1(p, v) * l1(w, s) * l2(q, r, t, u)
                                        + quarter * l1(p, v) * l1(w, t) * l2(q, r, s, u)
                                        + quarter * l1(p, v) * l1(w, u) * l2(q, r, t, s)
                                        + quarter * l1(q, s) * l1(r, t) * l2(p, w, u, v)
                                        - half * l1(q, s) * l1(r, u) * l2(p, w, t, v)
                                        + quarter * l1(q, s) * l1(r, v) * l2(p, w, t, u)
                                        + quarter * l1(q, s) * l1(w, t) * l2(p, r, v, u)
                                        + quarter * l1(q, s) * l1(w, u) * l2(p, r, t, v)
                                        - half * l1(q, s) * l1(w, v) * l2(p, r, t, u)
                                        - half * l1(q, t) * l1(r, s) * l2(p, w, u, v)
                                        + l1(q, t) * l1(r, u) * l2(p, w, s, v)
                                        - half * l1(q, t) * l1(r, v) * l2(p, w, s, u)
                                        - half * l1(q, t) * l1(w, s) * l2(p, r, v, u)
                                        - half * l1(q, t) * l1(w, u) * l2(p, r, s, v)
                                        + l1(q, t) * l1(w, v) * l2(p, r, s, u)
                                        + quarter * l1(q, u) * l1(r, s) * l2(p, w, t, v)
                                        - half * l1(q, u) * l1(r, t) * l2(p, w, s, v)
                                        + quarter * l1(q, u) * l1(r, v) * l2(p, w, s, t)
                                        + quarter * l1(q, u) * l1(w, s) * l2(p, r, v, t)
                                        + quarter * l1(q, u) * l1(w, t) * l2(p, r, s, v)
                                        - half * l1(q, u) * l1(w, v) * l2(p, r, s, t)
                                        + quarter * l1(q, v) * l1(r, s) * l2(p, w, u, t)
                                        + quarter * l1(q, v) * l1(r, t) * l2(p, w, s, u)
                                        - half * l1(q, v) * l1(r, u) * l2(p, w, s, t)
                                        + quarter * l1(q, v) * l1(w, s) * l2(p, r, t, u)
                                        - half * l1(q, v) * l1(w, t) * l2(p, r, s, u)
                                        + quarter * l1(q, v) * l1(w, u) * l2(p, r, s, t)
                                        + quarter * l1(r, s) * l1(w, t) * l2(p, q, u, v)
                                        + quarter * l1(r, s) * l1(w, u) * l2(p, q, v, t)
                                        - half * l1(r, s) * l1(w, v) * l2(p, q, u, t)
                                        + quarter * l1(r, t) * l1(w, s) * l2(p, q, v, u)
                                        + quarter * l1(r, t) * l1(w, u) * l2(p, q, s, v)
                                        - half * l1(r, t) * l1(w, v) * l2(p, q, s, u)
                                        - half * l1(r, u) * l1(w, s) * l2(p, q, v, t)
                                        - half * l1(r, u) * l1(w, t) * l2(p, q, s, v)
                                        + l1(r, u) * l1(w, v) * l2(p, q, s, t)
                                        + quarter * l1(r, v) * l1(w, s) * l2(p, q, u, t)
                                        + quarter * l1(r, v) * l1(w, t) * l2(p, q, s, u)
                                        - half * l1(r, v) * l1(w, u) * l2(p, q, s, t)
                                        + l1(p, s) * l3(q, r, w, t, u, v)
                                        - half * l1(p, t) * l3(q, r, w, s, u, v)
                                        - half * l1(p, u) * l3(q, r, w, t, s, v)
                                        - half * l1(p, v) * l3(q, r, w, t, u, s)
                                        - half * l1(q, s) * l3(p, r, w, t, u, v)
                                        + l1(q, t) * l3(p, r, w, s, u, v)
                                        - half * l1(q, u) * l3(p, r, w, s, t, v)
                                        - half * l1(q, v) * l3(p, r, w, s, u, t)
                                        - half * l1(r, s) * l3(p, q, w, u, t, v)
                                        - half * l1(r, t) * l3(p, q, w, s, u, v)
                                        + l1(r, u) * l3(p, q, w, s, t, v)
                                        - half * l1(r, v) * l3(p, q, w, s, t, u)
                                        - half * l1(w, s) * l3(p, q, r, v, t, u)
                                        - half * l1(w, t) * l3(p, q, r, s, v, u)
                                        - half * l1(w, u) * l3(p, q, r, s, t, v)
                                        + l1(w, v) * l3(p, q, r, s, t, u)
                                        + l2(p, q, s, t) * l2(r, w, u, v)
                                        - half * l2(p, q, s, u) * l2(r, w, t, v)
                                        - half * l2(p, q, s, v) * l2(r, w, u, t)
                                        - half * l2(p, q, u, t) * l2(r, w, s, v)
                                        + third * l2(p, q, u, v) * l2(r, w, s, t)
                                        + sixth * l2(p, q, u, v) * l2(r, w, t, s)
                                        - half * l2(p, q, v, t) * l2(r, w, u, s)
                                        + sixth * l2(p, q, v, u) * l2(r, w, s, t)
                                        + third * l2(p, q, v, u) * l2(r, w, t, s)
                                        - half * l2(p, r, s, t) * l2(q, w, u, v)
                                        + l2(p, r, s, u) * l2(q, w, t, v)
                                        - half * l2(p, r, s, v) * l2(q, w, t, u)
                                        - half * l2(p, r, t, u) * l2(q, w, s, v)
                                        + third * l2(p, r, t, v) * l2(q, w, s, u)
                                        + sixth * l2(p, r, t, v) * l2(q, w, u, s)
                                        + sixth * l2(p, r, v, t) * l2(q, w, s, u)
                                        + third * l2(p, r, v, t) * l2(q, w, u, s)
                                        - half * l2(p, r, v, u) * l2(q, w, t, s)
                                        - half * l2(p, w, s, t) * l2(q, r, v, u)
                                        - half * l2(p, w, s, u) * l2(q, r, t, v)
                                        + l2(p, w, s, v) * l2(q, r, t, u)
                                        + third * l2(p, w, t, u) * l2(q, r, s, v)
                                        + sixth * l2(p, w, t, u) * l2(q, r, v, s)
                                        - half * l2(p, w, t, v) * l2(q, r, s, u)
                                        + sixth * l2(p, w, u, t) * l2(q, r, s, v)
                                        + third * l2(p, w, u, t) * l2(q, r, v, s)
                                        - half * l2(p, w, u, v) * l2(q, r, t, s);

                                    let refv = gamma4.data[g4i] - disconnected;
                                    l4err = l4err.max((l4(p, q, r, w, s, t, u, v) - refv).abs());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("{}", "=".repeat(100));
    println!("NOCI spin-free cumulant diagnostics");
    println!("Max Lambda1 - active Gamma1 error: {:.6e}", l1err);
    println!(
        "Max Lambda2 explicit spin-free formula error: {:.6e}",
        l2err
    );
    println!(
        "Max Lambda3 explicit spin-free formula error: {:.6e}",
        l3err
    );
    println!(
        "Max Lambda4 explicit spin-free formula error: {:.6e}",
        l4err
    );

    println!(
        "Max |Lambda1|: {:.6e}",
        lambda
            .lambda1
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
    );
    println!(
        "Max |Lambda2|: {:.6e}",
        lambda
            .lambda2
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
    );
    println!(
        "Max |Lambda3|: {:.6e}",
        lambda
            .lambda3
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
    );
    println!(
        "Max |Lambda4|: {:.6e}",
        lambda
            .lambda4
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
    );
}
