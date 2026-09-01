scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
}

mol = {
    basis = "STO-3G",
    r = {1.5},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("H 0 0 %g", -2.5 * r),
            string.format("H 0 0 %g", -1.5 * r),
            string.format("H 0 0 %g", -0.5 * r),
            string.format("H 0 0 %g",  0.5 * r),
            string.format("H 0 0 %g",  1.5 * r),
            string.format("H 0 0 %g",  2.5 * r),
        }
    end,
}

states = {
    mom = {
        {label = "RHF", noci = true},
        {
            label = "UHF (+-+-+-)",
            spin_bias = {pattern = {1, -1, 1, -1, 1, -1}, pol = 0.75},
            noci = true,
        },
        {
            label = "UHF (-+-+-+)",
            spin_bias = {pattern = {-1, 1, -1, 1, -1, 1}, pol = 0.75},
            noci = true,
        },
    },
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}

excit = {
    orders = {1, 2, 3},
}

prop = {
    dt = 1e-4,
    propagator = "direct-overlap",
}

qmc = {
    initial_population = 10000,
    target_population = 1e9,
    ncycles = 1,
    nreports = 5,
    sampling_cutoff1 = 0.0,
    sampling_cutoff2 = 0.0,
    spawn_cutoff = 0.0,
    excitation_gen = "overlap-weighted",
    factor_tables = "ram",
    overlap_weight = 0.5,
    optimise_overlap_weight = true,
    seed = 2,
}
