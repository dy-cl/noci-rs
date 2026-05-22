// input/parse.rs

use std::fs;

use rlua::{Lua, Table, Value};

use super::{
    DeterministicOptions, DiisOptions, ExcitationGen, ExcitationOptions, GMRESOptions, HSCFOptions,
    Input, Metadynamics, MolOptions, PropagationOptions, Propagator, QMCOptions, SCFExcitation,
    SCFInfo, SNOCIOptions, SNOCIPreconditioner, SpatialBias, Spin, SpinBias, StateRecipe,
    StateType, WicksOptions, WicksStorage, WriteOptions,
};

/// Read required table from Lua globals.
/// # Arguments:
/// - `globals`: Lua globals table.
/// - `name`: Required table name.
/// # Returns:
/// - `Table`: Parsed Lua table.
fn required_table<'lua>(
    globals: &Table<'lua>,
    name: &str,
) -> Table<'lua> {
    globals.get(name).unwrap_or_else(|_| {
        println!("Missing required table '{name}'");
        std::process::exit(1);
    })
}

/// Read integer pattern entries taking values in {-1, 0, 1}.
/// # Arguments:
/// - `pat_tbl`: Lua table containing the pattern entries.
/// # Returns:
/// - `Vec<i8>`: Parsed pattern entries.
fn read_pattern(pat_tbl: Table) -> Vec<i8> {
    pat_tbl
        .sequence_values::<i64>()
        .map(|x| x.unwrap())
        .map(|x| match x {
            -1 => -1,
            0 => 0,
            1 => 1,
            _ => {
                eprintln!("pattern entries must be -1, 0, or 1");
                std::process::exit(1);
            }
        })
        .collect()
}

/// Read basis state recipe from Lua table.
/// # Arguments:
/// - `t`: Lua table containing the state recipe specification.
/// # Returns:
/// - `StateRecipe`: Parsed state recipe with optional spin bias, spatial bias, and SCF excitation data.
fn read_state_recipe(t: Table) -> StateRecipe {
    let defaults = StateRecipe::default();
    let label: String = t.get("label").unwrap_or(defaults.label);
    let noci: bool = t.get("noci").unwrap_or(defaults.noci);
    let holomorphic: bool = t.get("holomorphic").unwrap_or(defaults.holomorphic);
    let partner: Option<String> = t.get("partner").unwrap_or(defaults.partner);

    let spin_bias = t
        .get::<_, Option<Table>>("spin_bias")
        .unwrap_or(None)
        .map(|sb| {
            let defaults = SpinBias::default();
            let pol: f64 = sb.get("pol").unwrap_or(defaults.pol);
            let pat_tbl: Table = sb.get("pattern").unwrap();
            let pattern = read_pattern(pat_tbl);
            SpinBias { pattern, pol }
        });

    let spatial_bias = t
        .get::<_, Option<Table>>("spatial_bias")
        .unwrap_or(None)
        .map(|sb| {
            let defaults = SpatialBias::default();
            let pol: f64 = sb.get("pol").unwrap_or(defaults.pol);
            let pat_tbl: Table = sb.get("pattern").unwrap();
            let pattern = read_pattern(pat_tbl);
            SpatialBias { pattern, pol }
        });

    let scfexcitation = t
        .get::<_, Option<Table>>("excit")
        .unwrap_or(None)
        .map(|ex| {
            let defaults = SCFExcitation::default();
            let s: String = ex
                .get("spin")
                .unwrap_or_else(|_| defaults.spin.as_str().to_string());
            let spin: Spin = s.parse().unwrap_or_else(|msg| {
                eprintln!("{msg}");
                std::process::exit(1);
            });
            SCFExcitation {
                spin,
                occ: ex.get("occ").unwrap_or(defaults.occ),
                vir: ex.get("vir").unwrap_or(defaults.vir),
            }
        });

    StateRecipe {
        label,
        spin_bias,
        spatial_bias,
        scfexcitation,
        partner,
        noci,
        holomorphic,
    }
}

/// Read molecular options from Lua table.
/// # Arguments:
/// - `mol_tbl`: Lua mol table.
/// # Returns:
/// - `MolOptions`: Parsed molecular options.
fn read_mol(mol_tbl: Table) -> MolOptions {
    let basis: String = mol_tbl.get("basis").unwrap();
    let unit: String = mol_tbl.get("unit").unwrap();
    let r_val: Value = mol_tbl.get("r").unwrap();
    let mut r_list: Vec<f64> = Vec::new();
    match r_val {
        Value::Number(x) => r_list.push(x),
        Value::Table(t) => {
            for item in t.sequence_values::<f64>() {
                let r = item.unwrap();
                r_list.push(r);
            }
        }
        _ => {
            eprintln!("Number or table required by mol.r");
            std::process::exit(1);
        }
    }
    let atoms_val: Value = mol_tbl.get("atoms").unwrap();
    let geoms: Vec<Vec<String>> = match atoms_val {
        Value::Table(t) => {
            let static_atoms = t
                .sequence_values::<String>()
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();
            vec![static_atoms; r_list.len()]
        }
        Value::Function(f) => {
            let mut out = Vec::with_capacity(r_list.len());
            for &r in &r_list {
                let tbl: Table = f.call(r).unwrap();
                let atoms = tbl
                    .sequence_values::<String>()
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>();
                out.push(atoms);
            }
            out
        }
        _ => {
            eprintln!("Table or function required by mol.atoms");
            std::process::exit(1);
        }
    };
    MolOptions {
        basis,
        unit,
        r_list,
        geoms,
    }
}

/// Read SCF options from optional Lua table.
/// # Arguments:
/// - `scf_tbl`: Optional Lua scf table.
/// # Returns:
/// - `SCFInfo`: Parsed SCF options.
fn read_scf(scf_tbl: Option<Table>) -> SCFInfo {
    if let Some(scf_tbl) = scf_tbl {
        let defaults = SCFInfo::default();
        let diis_defaults = DiisOptions::default();
        let h_defaults = HSCFOptions::default();
        let diis_tbl: Option<Table> = scf_tbl.get::<_, Option<Table>>("diis").unwrap_or(None);
        let h_tbl: Option<Table> = scf_tbl.get::<_, Option<Table>>("h").unwrap_or(None);
        let diis = if let Some(diis_tbl) = diis_tbl {
            DiisOptions {
                space: diis_tbl.get("space").unwrap_or(diis_defaults.space),
            }
        } else {
            diis_defaults
        };
        let h = if let Some(h_tbl) = h_tbl {
            HSCFOptions {
                max_cycle: h_tbl.get("max_cycle").unwrap_or(h_defaults.max_cycle),
                g_tol: h_tbl.get("g_tol").unwrap_or(h_defaults.g_tol),
                sr1_tol: h_tbl.get("sr1_tol").unwrap_or(h_defaults.sr1_tol),
                denom_tol: h_tbl.get("denom_tol").unwrap_or(h_defaults.denom_tol),
                max_step: h_tbl.get("max_step").unwrap_or(h_defaults.max_step),
                line_steps: h_tbl.get("line_steps").unwrap_or(h_defaults.line_steps),
                line_shrink: h_tbl.get("line_shrink").unwrap_or(h_defaults.line_shrink),
                history: h_tbl.get("history").unwrap_or(h_defaults.history),
            }
        } else {
            h_defaults
        };
        SCFInfo {
            max_cycle: scf_tbl.get("max_cycle").unwrap_or(defaults.max_cycle),
            e_tol: scf_tbl.get("e_tol").unwrap_or(defaults.e_tol),
            fds_sdf_tol: scf_tbl.get("fds_sdf_tol").unwrap_or(defaults.fds_sdf_tol),
            d_tol: scf_tbl.get("d_tol").unwrap_or(defaults.d_tol),
            diis,
            do_fci: scf_tbl.get("do_fci").unwrap_or(defaults.do_fci),
            h,
        }
    } else {
        SCFInfo::default()
    }
}

/// Read write options from optional Lua table.
/// # Arguments:
/// - `write_tbl`: Optional Lua write table.
/// # Returns:
/// - `WriteOptions`: Parsed write options.
fn read_write(write_tbl: Option<Table>) -> WriteOptions {
    if let Some(write_tbl) = write_tbl {
        let defaults = WriteOptions::default();
        WriteOptions {
            verbose: write_tbl.get("verbose").unwrap_or(defaults.verbose),
            write_deterministic_coeffs: write_tbl
                .get("write_deterministic_coeffs")
                .unwrap_or(defaults.write_deterministic_coeffs),
            write_orbitals: write_tbl
                .get("write_orbitals")
                .unwrap_or(defaults.write_orbitals),
            write_excitation_hist: write_tbl
                .get("write_excitation_hist")
                .unwrap_or(defaults.write_excitation_hist),
            write_matrices: write_tbl
                .get("write_matrices")
                .unwrap_or(defaults.write_matrices),
            write_dir: write_tbl.get("write_dir").unwrap_or(defaults.write_dir),
            write_restart: write_tbl
                .get("write_restart")
                .unwrap_or(defaults.write_restart),
            read_restart: write_tbl
                .get("read_restart")
                .unwrap_or(defaults.read_restart),
        }
    } else {
        WriteOptions::default()
    }
}

/// Read state search options from Lua table.
/// # Arguments:
/// - `state_tbl`: Lua states table.
/// # Returns:
/// - `StateType`: Parsed state search options.
fn read_states(state_tbl: Table) -> StateType {
    let mom_tbl: Option<Table> = state_tbl.get::<_, Option<Table>>("mom").unwrap_or(None);
    let meta_tbl: Option<Table> = state_tbl
        .get::<_, Option<Table>>("metadynamics")
        .unwrap_or(None);

    match (mom_tbl, meta_tbl) {
        (Some(_), Some(_)) => {
            eprintln!("Cannot use MOM and SCF metadynamics simultaneously.");
            std::process::exit(1);
        }
        (Some(mom_tbl), None) => {
            let mut recipes: Vec<StateRecipe> = Vec::new();
            for st in mom_tbl.sequence_values::<rlua::Table>() {
                let t = st.unwrap();
                recipes.push(read_state_recipe(t));
            }
            StateType::Mom(recipes)
        }
        (None, Some(meta_tbl)) => {
            let defaults = Metadynamics::default();
            let nstates_rhf: usize = meta_tbl.get("nstates_rhf").unwrap_or(defaults.nstates_rhf);
            let nstates_uhf: usize = meta_tbl.get("nstates_uhf").unwrap_or(defaults.nstates_uhf);
            let spinpol: f64 = meta_tbl.get("spinpol").unwrap_or(defaults.spinpol);
            let spatialpol: f64 = meta_tbl.get("spatialpol").unwrap_or(defaults.spatialpol);
            let lambda: f64 = meta_tbl.get("lambda").unwrap_or(defaults.lambda);
            let max_attempts: usize = meta_tbl
                .get("max_attempts")
                .unwrap_or(defaults.max_attempts);

            let labels_rhf = (1..=nstates_rhf)
                .map(|k| format!("RHF M {}", k))
                .collect::<Vec<_>>();
            let labels_uhf = (1..=nstates_uhf)
                .map(|k| {
                    let pair = k.div_ceil(2);
                    let ab = if (k % 2) == 1 { "A" } else { "B" };
                    format!("UHF M {} {}", pair, ab)
                })
                .collect::<Vec<_>>();
            let spatial_patterns_rhf = vec![None; nstates_rhf];
            let spin_patterns_uhf = vec![None; nstates_uhf];

            StateType::Metadynamics(Metadynamics {
                nstates_rhf,
                nstates_uhf,
                spinpol,
                spatialpol,
                lambda,
                labels_rhf,
                labels_uhf,
                spatial_patterns_rhf,
                spin_patterns_uhf,
                max_attempts,
            })
        }
        (None, None) => {
            eprintln!("Must use either MOM or SCF metadynamics to locate SCF solutions");
            std::process::exit(1);
        }
    }
}

/// Read deterministic options from optional Lua table.
/// # Arguments:
/// - `det_tbl`: Optional Lua det table.
/// # Returns:
/// - `Option<DeterministicOptions>`: Parsed deterministic options.
fn read_det(det_tbl: Option<Table>) -> Option<DeterministicOptions> {
    det_tbl.map(|det_tbl| {
        let defaults = DeterministicOptions::default();
        DeterministicOptions {
            max_steps: det_tbl.get("max_steps").unwrap_or(defaults.max_steps),
            dynamic_shift: det_tbl
                .get("dynamic_shift")
                .unwrap_or(defaults.dynamic_shift),
            dynamic_shift_alpha: det_tbl
                .get("dynamic_shift_alpha")
                .unwrap_or(defaults.dynamic_shift_alpha),
            e_tol: det_tbl.get("e_tol").unwrap_or(defaults.e_tol),
        }
    })
}

/// Read QMC options from optional Lua table.
/// # Arguments:
/// - `qmc_tbl`: Optional Lua qmc table.
/// # Returns:
/// - `Option<QMCOptions>`: Parsed QMC options.
fn read_qmc(qmc_tbl: Option<Table>) -> Option<QMCOptions> {
    qmc_tbl.map(|qmc_tbl| {
        let defaults = QMCOptions::default();
        let excitation_gen_str: String =
            qmc_tbl
                .get("excitation_gen")
                .unwrap_or_else(|_| match defaults.excitation_gen {
                    ExcitationGen::Uniform => "uniform".to_string(),
                    ExcitationGen::HeatBath => "heat-bath".to_string(),
                    ExcitationGen::ApproximateHeatBath => "approximate-heat-bath".to_string(),
                });
        let excitation_gen: ExcitationGen = excitation_gen_str.parse().unwrap_or_else(|msg| {
            eprintln!("{msg}");
            std::process::exit(1);
        });
        QMCOptions {
            initial_population: qmc_tbl
                .get("initial_population")
                .unwrap_or(defaults.initial_population),
            target_population: qmc_tbl
                .get("target_population")
                .unwrap_or(defaults.target_population),
            shift_damping: qmc_tbl
                .get("shift_damping")
                .unwrap_or(defaults.shift_damping),
            ncycles: qmc_tbl.get("ncycles").unwrap_or(defaults.ncycles),
            nreports: qmc_tbl.get("nreports").unwrap_or(defaults.nreports),
            excitation_gen,
            seed: qmc_tbl.get("seed").unwrap_or(defaults.seed),
        }
    })
}

/// Read SNOCI options from optional Lua table.
/// # Arguments:
/// - `snoci_tbl`: Optional Lua snoci table.
/// # Returns:
/// - `Option<SNOCIOptions>`: Parsed SNOCI options.
fn read_snoci(snoci_tbl: Option<Table>) -> Option<SNOCIOptions> {
    snoci_tbl.map(|snoci_tbl| {
        let defaults = SNOCIOptions::default();
        let gmres_defaults = GMRESOptions::default();

        let gmres_tbl: Option<Table> = snoci_tbl.get::<_, Option<Table>>("gmres").unwrap_or(None);

        let gmres = if let Some(gmres_tbl) = gmres_tbl {
            GMRESOptions {
                max_iter: gmres_tbl.get("max_iter").unwrap_or(gmres_defaults.max_iter),
                res_tol: gmres_tbl.get("res_tol").unwrap_or(gmres_defaults.res_tol),
                metric_tol: gmres_tbl
                    .get("metric_tol")
                    .unwrap_or(gmres_defaults.metric_tol),
                restart: gmres_tbl.get("restart").unwrap_or(gmres_defaults.restart),
                full_m: gmres_tbl.get("full_m").unwrap_or(gmres_defaults.full_m),
            }
        } else {
            gmres_defaults
        };

        let preconditioner_str: String = snoci_tbl
            .get("preconditioner")
            .unwrap_or_else(|_| defaults.preconditioner.as_str().to_string());

        let preconditioner: SNOCIPreconditioner =
            preconditioner_str.parse().unwrap_or_else(|msg| {
                eprintln!("{msg}");
                std::process::exit(1);
            });

        let imag_shifts: Vec<f64> = snoci_tbl
            .get::<_, Option<Table>>("imag_shift")
            .unwrap_or(None)
            .map(|t| {
                t.sequence_values::<f64>()
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(defaults.imag_shifts.clone());

        SNOCIOptions {
            sigma: snoci_tbl.get("sigma").unwrap_or(defaults.sigma),
            tol: snoci_tbl.get("tol").unwrap_or(defaults.tol),
            imag_shifts,
            max_iter: snoci_tbl.get("max_iter").unwrap_or(defaults.max_iter),
            max_add: snoci_tbl.get("max_add").unwrap_or(defaults.max_add),
            max_dim: snoci_tbl.get("max_dim").unwrap_or(defaults.max_dim),
            preconditioner,
            gmres,
        }
    })
}

/// Read excitation options from optional Lua table.
/// # Arguments:
/// - `excit_tbl`: Optional Lua excit table.
/// # Returns:
/// - `ExcitationOptions`: Parsed excitation options.
fn read_excit(excit_tbl: Option<Table>) -> ExcitationOptions {
    if let Some(excit_tbl) = excit_tbl {
        let defaults = ExcitationOptions::default();

        let orders: Option<Vec<usize>> = match excit_tbl.get::<_, Value>("orders") {
            Ok(Value::Nil) => None,
            Ok(_) => Some(excit_tbl.get("orders").unwrap_or_else(|msg| {
                eprintln!("{msg}");
                std::process::exit(1);
            })),
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        let all: bool = match excit_tbl.get::<_, Value>("all") {
            Ok(Value::Nil) => defaults.all,
            Ok(_) => excit_tbl.get("all").unwrap_or_else(|msg| {
                eprintln!("{msg}");
                std::process::exit(1);
            }),
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        if all && orders.is_some() {
            eprintln!("Cannot specify both excit.orders and excit.all = true");
            std::process::exit(1);
        }

        ExcitationOptions {
            orders: if all {
                Vec::new()
            } else {
                orders.unwrap_or(defaults.orders)
            },
            all,
        }
    } else {
        ExcitationOptions::default()
    }
}

/// Read propagation options from optional Lua table.
/// # Arguments:
/// - `prop_tbl`: Optional Lua prop table.
/// # Returns:
/// - `Option<PropagationOptions>`: Parsed propagation options.
fn read_prop(prop_tbl: Option<Table>) -> Option<PropagationOptions> {
    prop_tbl.map(|prop_tbl| {
        let defaults = PropagationOptions::default();

        let propagator_str: String = prop_tbl
            .get("propagator")
            .unwrap_or_else(|_| defaults.propagator.as_str().to_string());
        let propagator: Propagator = propagator_str.parse().unwrap_or_else(|msg| {
            eprintln!("{msg}");
            std::process::exit(1);
        });
        PropagationOptions {
            dt: prop_tbl.get("dt").unwrap_or(defaults.dt),
            propagator,
        }
    })
}

/// Read Wick's theorem options from optional Lua table.
/// # Arguments:
/// - `wicks_tbl`: Optional Lua wicks table.
/// # Returns:
/// - `WicksOptions`: Parsed Wick's theorem options.
fn read_wicks(wicks_tbl: Option<Table>) -> WicksOptions {
    if let Some(wicks_tbl) = wicks_tbl {
        let defaults = WicksOptions::default();
        let storage = match wicks_tbl.get::<_, Option<String>>("storage").unwrap() {
            Some(s) => match s.to_lowercase().as_str() {
                "ram" => WicksStorage::RAM,
                "disk" => WicksStorage::Disk,
                other => panic!("Unknown wicks.storage value: {other}. Use 'ram' or 'disk'."),
            },
            None => defaults.storage,
        };
        WicksOptions {
            compare: wicks_tbl.get("compare").unwrap_or(defaults.compare),
            enabled: wicks_tbl.get("enabled").unwrap_or(defaults.enabled),
            storage,
            cachedir: wicks_tbl.get("cachedir").unwrap_or(defaults.cachedir),
        }
    } else {
        WicksOptions::default()
    }
}

/// Read input parameters from lua file and assign to Input object.
/// # Arguments
/// - `path`: File path to input file.
/// # Returns:
/// - `Input`: Parsed input options.
pub fn load_input(path: &str) -> Input {
    let src = fs::read_to_string(path).unwrap();
    let lua = Lua::new();

    let ctx = lua;
    ctx.load(&src).exec().unwrap();
    let globals = ctx.globals();

    let mol_tbl = required_table(&globals, "mol");
    let state_tbl = required_table(&globals, "states");
    let scf_tbl: Option<Table> = globals.get::<_, Option<Table>>("scf").unwrap_or(None);
    let write_tbl: Option<Table> = globals.get::<_, Option<Table>>("write").unwrap_or(None);
    let excit_tbl: Option<Table> = globals.get::<_, Option<Table>>("excit").unwrap_or(None);
    let prop_tbl: Option<Table> = globals.get::<_, Option<Table>>("prop").unwrap_or(None);
    let wicks_tbl: Option<Table> = globals.get::<_, Option<Table>>("wicks").unwrap_or(None);
    let det_tbl: Option<Table> = globals.get::<_, Option<Table>>("det").unwrap_or(None);
    let qmc_tbl: Option<Table> = globals.get::<_, Option<Table>>("qmc").unwrap_or(None);
    let snoci_tbl: Option<Table> = globals.get::<_, Option<Table>>("snoci").unwrap_or(None);

    Input {
        mol: read_mol(mol_tbl),
        scf: read_scf(scf_tbl),
        write: read_write(write_tbl),
        states: read_states(state_tbl),
        det: read_det(det_tbl),
        qmc: read_qmc(qmc_tbl),
        snoci: read_snoci(snoci_tbl),
        excit: read_excit(excit_tbl),
        prop: read_prop(prop_tbl),
        wicks: read_wicks(wicks_tbl),
    }
}
