scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
}

mol = {
    basis = 'STO-3G',
    r = {2.8},
    unit = 'Ang',
    atoms = function(r)
        return {string.format("Li 0 0 %g", -r / 2), string.format("H 0 0 %g",  r / 2),}
        end,
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = false},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

wicks = {
    enabled = false,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}

excit = {
    orders = {1, 2},
}

prop = {
    dt = 1e-2,
    propagator = "direct-overlap",
}

qmc = {
    initial_population = 200,
    target_population = 500,
    ncycles = 1,
    nreports = 15,
    fri = {
        population = { cutoff = 1.0 },
        spawn = { cutoff = 0.25 },
        pre_overlap = { target_nnz = 1000000000 },
        shift_tangent = { target_nnz = 1000000000 },
    },
    excitation_gen = "uniform",
    seed = 1,
}
