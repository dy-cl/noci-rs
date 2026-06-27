scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = 'cc-pVDZ',
    r = {1.5},
    unit = 'Ang',
    atoms = function(r)
        return {string.format("H 0 0 %g", -r / 2), string.format("H 0 0 %g",  r / 2),}
        end,
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = true},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}

excit = {
    orders = {1, 2},
}

prop = {
    dt = 1e-4,
    propagator = "difference-doubly-shifted-u2",
}

qmc = {
    initial_population = 5e2,
    target_population = 1e4,
    ncycles = 1e1,
    nreports = 1e3,
    excitation_gen = "uniform",
    seed = 1,
}

