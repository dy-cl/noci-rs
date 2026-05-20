// write.rs
use std::fmt::Display;
use std::fs::{File as StdFile, create_dir_all};
use std::io::{BufWriter, Write};

use hdf5::File;
use hdf5::types::VarLenUnicode;

use ndarray::{Array1, Array2};

use crate::AoData;
use crate::input::{ExcitationGen, Input, Spin, StateType, WicksStorage};

/// Print input options at the top of output.
/// # Arguments:
/// - `input`: User specified input options.
/// # Returns
/// - `()`: Prints the parsed input options to stdout.
pub fn print_input(input: &Input) {
    let left = "=".repeat(45);
    let right = "=".repeat(46);
    println!("{}INPUT OPTIONS{}", left, right);

    println!("MOL");
    println!("BASIS: {}", input.mol.basis);
    println!("UNIT: {}", input.mol.unit);
    println!("R: {:?}", input.mol.r_list);
    println!("NGEOMS: {}", input.mol.geoms.len());
    if let Some(g0) = input.mol.geoms.first() {
        println!("GEOM[0]: {:?}", g0);
    }
    println!();

    println!("SCF");
    println!("MAX_CYCLE: {}", input.scf.max_cycle);
    println!("ETOL: {}", input.scf.e_tol);
    println!("FDS_SDF_TOL: {}", input.scf.fds_sdf_tol);
    println!("DTOL: {}", input.scf.d_tol);
    println!("DIIS SPACE DIM: {}", input.scf.diis.space);
    println!("DO_FCI: {}", input.scf.do_fci);
    println!();

    println!("HSCF");
    println!("  MAX_CYCLE: {}", input.scf.h.max_cycle);
    println!("  G_TOL: {:.12e}", input.scf.h.g_tol);
    println!("  SR1_TOL: {:.12e}", input.scf.h.sr1_tol);
    println!("  DENOM_TOL: {:.12e}", input.scf.h.denom_tol);
    println!("  MAX_STEP: {:.12e}", input.scf.h.max_step);
    println!("  LINE_STEPS: {}", input.scf.h.line_steps);
    println!("  LINE_SHRINK: {:.12e}", input.scf.h.line_shrink);
    println!("  HISTORY: {}", input.scf.h.history);

    if let StateType::Mom(recipes) = &input.states {
        for recipe in recipes {
            println!("  LABEL: {}", recipe.label);
            println!("    HOLOMORPHIC: {}", recipe.holomorphic);
            println!("    PARTNER: {:?}", recipe.partner);
        }
    }
    println!();

    println!("STATES");
    match &input.states {
        StateType::Mom(recipes) => {
            println!("TYPE: MOM");
            println!("NRECIPES: {}", recipes.len());
            for (i, r) in recipes.iter().enumerate() {
                println!("RECIPE {}:", i + 1);
                println!("  LABEL: {}", r.label);
                println!("  NOCI: {}", r.noci);

                match &r.spin_bias {
                    Some(sb) => {
                        println!("  SPIN_BIAS:");
                        println!("    POL: {:.6}", sb.pol);
                        println!("    PATTERN: {:?}", sb.pattern);
                    }
                    None => println!("  SPIN_BIAS: NONE"),
                }

                match &r.spatial_bias {
                    Some(sb) => {
                        println!("  SPATIAL_BIAS:");
                        println!("    POL: {:.6}", sb.pol);
                        println!("    PATTERN: {:?}", sb.pattern);
                    }
                    None => println!("  SPATIAL_BIAS: NONE"),
                }

                match &r.scfexcitation {
                    Some(ex) => {
                        println!("  EXCIT:");
                        let spin = match ex.spin {
                            Spin::Alpha => "alpha",
                            Spin::Beta => "beta",
                            Spin::Both => "both",
                        };
                        println!("    SPIN: {}", spin);
                        println!("    OCC: {}", ex.occ);
                        println!("    VIR: {}", ex.vir);
                    }
                    None => println!("  EXCIT: NONE"),
                }
            }
        }
        StateType::Metadynamics(m) => {
            println!("TYPE: METADYNAMICS");
            println!("NSTATES_RHF: {}", m.nstates_rhf);
            println!("NSTATES_UHF: {}", m.nstates_uhf);
            println!("SPINPOL: {:.6}", m.spinpol);
            println!("SPATIALPOL: {:.6}", m.spatialpol);
            println!("LAMBDA: {:.6}", m.lambda);
            println!("MAX_ATTEMPTS: {}", m.max_attempts);
        }
    }
    println!();

    println!("EXCIT");
    println!("LEVEL (ORDER): {:?}", input.excit.orders);
    println!("ALL: {}", input.excit.all);
    println!();

    println!("PROP");
    if let Some(prop) = input.prop.as_ref() {
        println!("DT: {}", prop.dt);
        println!("PROPAGATOR: {}", prop.propagator.as_str());
    } else {
        println!("NONE");
    }
    println!();

    println!("DET");
    match &input.det {
        Some(d) => {
            println!("ENABLED: true");
            println!("MAX_STEPS: {}", d.max_steps);
            println!("DYNAMIC_SHIFT: {}", d.dynamic_shift);
            println!("DYNAMIC_SHIFT_ALPHA: {}", d.dynamic_shift_alpha);
            println!("ETOL: {:}", d.e_tol);
        }
        None => {
            println!("ENABLED: false");
        }
    }
    println!();

    println!("QMC");
    match &input.qmc {
        Some(q) => {
            println!("ENABLED: true");
            println!("INITIAL_POPULATION: {}", q.initial_population);
            println!("TARGET_POPULATION: {}", q.target_population);
            println!("NCYCLES: {}", q.ncycles);
            println!("NREPORTS: {}", q.nreports);
            println!("SHIFT_DAMPING: {}", q.shift_damping);
            let excitation_gen = match q.excitation_gen {
                ExcitationGen::Uniform => "uniform",
                ExcitationGen::HeatBath => "heat-bath",
                ExcitationGen::ApproximateHeatBath => "approximate-heat-bath",
            };
            println!("EXCITATION_GEN: {}", excitation_gen);
            println!("SEED: {:?}", q.seed);
        }
        None => {
            println!("ENABLED: false");
        }
    }
    println!();

    println!("SNOCI");
    match &input.snoci {
        Some(s) => {
            println!("ENABLED: true");
            println!("MAX_ITER: {}", s.max_iter);
            println!("MAX_DIM: {}", s.max_dim);
            println!("MAX_ADD: {}", s.max_add);
            println!("SIGMA: {}", s.sigma);
            println!("TOL: {}", s.tol);
            println!("IMAG_SHIFTS: {:?}", s.imag_shifts);
            println!("PRECONDITIONER: {}", s.preconditioner.as_str());
            println!("GMRES:");
            println!("  MAX_ITER: {}", s.gmres.max_iter);
            println!("  RESTART: {}", s.gmres.restart);
            println!("  RES_TOL: {}", s.gmres.res_tol);
            println!("  METRIC_TOL: {}", s.gmres.metric_tol);
            println!("  FULL_M: {}", s.gmres.full_m);
        }
        None => {
            println!("ENABLED: false");
        }
    }
    println!();

    println!("WRITE");
    println!("VERBOSE: {}", input.write.verbose);
    println!("WRITE_DIR: {}", input.write.write_dir);
    println!(
        "WRITE_DETERMINISTIC_COEFFS: {}",
        input.write.write_deterministic_coeffs
    );
    println!("WRITE_ORBITALS: {}", input.write.write_orbitals);
    println!(
        "WRITE_EXCITATION_HIST: {}",
        input.write.write_excitation_hist
    );
    println!("WRITE_MATRICES: {}", input.write.write_matrices);
    println!("WRITE_RESTART: {:?}", input.write.write_restart);
    println!("READ_RESTART: {:?}", input.write.read_restart);
    println!();

    println!("WICKS");
    println!("ENABLED: {}", input.wicks.enabled);
    println!("COMPARE: {}", input.wicks.compare);
    let storage = match &input.wicks.storage {
        WicksStorage::RAM => "ram",
        WicksStorage::Disk => "disk",
    };
    println!("STORAGE: {}", storage);
    println!("CACHEDIR: {:?}", input.wicks.cachedir);

    println!("{}", "=".repeat(100));
}

/// Write sufficient data of an SCF state into an HDF5  file such that we can plot the orbitals in post.
/// Of course, this does not contain the geometry or basis used, but this can be provided in the input to a
/// plotting script provided we have not forgotten what was used.
/// # Arguments:
/// - `path`: Filepath for the HDF5 file.
/// - `label`: Sanitised label of the SCF state.
/// - `ao`: Contains AO integrals and metadata.
/// - `ca`: MO coefficients spin alpha.  
/// - `cb`: MO coefficients spin beta.
/// - `ea`: MO energies spin alpha.
/// - `eb`: MO energies spin beta.
/// - `oa`: MO occupancies spin alpha.
/// - `ob`: MO occupancies spin beta.
/// - `da`: Spin alpha density matrix.
/// - `db`: Spin beta density matrix.
/// # Returns
/// - `()`: Writes orbital data to the HDF5 file at `path`.
pub fn write_orbitals(
    path: &str,
    ao: &AoData,
    label: &str,
    c: (&Array2<f64>, &Array2<f64>),
    e: (&Array1<f64>, &Array1<f64>),
    occ: (&[f64], &[f64]),
    d: (&Array2<f64>, &Array2<f64>),
) {
    let (ca, cb) = c;
    let (ea, eb) = e;
    let (oa, ob) = occ;
    let (da, db) = d;

    let f = File::create(path).unwrap();

    let vlabel: VarLenUnicode = label.parse().unwrap();
    f.new_dataset::<VarLenUnicode>()
        .create("label")
        .unwrap()
        .write_scalar(&vlabel)
        .unwrap();
    let labelsvlu: Vec<VarLenUnicode> = ao.labels.iter().map(|s| s.parse().unwrap()).collect();
    f.new_dataset::<VarLenUnicode>()
        .shape(labelsvlu.len())
        .create("aolabels")
        .unwrap()
        .write(&labelsvlu)
        .unwrap();

    let neleci64: Vec<i64> = ao.nelec.iter().copied().collect();
    f.new_dataset::<i64>()
        .shape(neleci64.len())
        .create("nelec")
        .unwrap()
        .write(&neleci64)
        .unwrap();

    let n = ao.s.ncols();
    f.new_dataset::<f64>()
        .shape((n, n))
        .create("S")
        .unwrap()
        .write(&ao.s)
        .unwrap();
    f.new_dataset::<f64>()
        .shape((n, n))
        .create("ca")
        .unwrap()
        .write(ca)
        .unwrap();
    f.new_dataset::<f64>()
        .shape((n, n))
        .create("cb")
        .unwrap()
        .write(cb)
        .unwrap();
    f.new_dataset::<f64>()
        .shape((n, n))
        .create("da")
        .unwrap()
        .write(da)
        .unwrap();
    f.new_dataset::<f64>()
        .shape((n, n))
        .create("db")
        .unwrap()
        .write(db)
        .unwrap();
    f.new_dataset::<f64>()
        .shape(ea.len())
        .create("ea")
        .unwrap()
        .write(ea)
        .unwrap();
    f.new_dataset::<f64>()
        .shape(eb.len())
        .create("eb")
        .unwrap()
        .write(eb)
        .unwrap();
    f.new_dataset::<f64>()
        .shape(oa.len())
        .create("oa")
        .unwrap()
        .write(oa)
        .unwrap();
    f.new_dataset::<f64>()
        .shape(ob.len())
        .create("ob")
        .unwrap()
        .write(ob)
        .unwrap();
}

/// Write matrix to disk.
/// # Arguments:
/// - `path`: Output path.
/// - `m`: Matrix to write.
/// # Returns:
/// - `()`: Writes matrix to disk.
pub fn write_matrix<T: Display>(
    path: &str,
    m: &Array2<T>,
) {
    let mut f = BufWriter::new(StdFile::create(path).unwrap());

    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if j > 0 {
                write!(f, " ").unwrap();
            }
            write!(f, "{}", m[(i, j)]).unwrap();
        }
        writeln!(f).unwrap();
    }
}

/// Write Hamiltonian and overlap matrices to disk.
/// # Arguments:
/// - `write_dir`: Directory to write matrices into.
/// - `h`: Hamiltonian matrix.
/// - `s`: Overlap matrix.
/// # Returns:
/// - `()`: Writes Hamiltonian and overlap matrices.
pub fn write_hs_matrices<T: Display>(
    write_dir: &str,
    h: &Array2<T>,
    s: &Array2<T>,
) {
    create_dir_all(write_dir).unwrap();
    write_matrix(&format!("{}/HAMI", write_dir), h);
    write_matrix(&format!("{}/OVLP", write_dir), s);
}
