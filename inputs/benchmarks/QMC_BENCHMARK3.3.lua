scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
}

mol = {
    basis = 'cc-pVDZ',
    r = {1.5},
    unit = 'Ang',
    atoms = function(r)
        local zs = {-2.5 * r, -1.5 * r, -0.5 * r, 0.5 * r, 1.5 * r, 2.5 * r}
        return {
            string.format("H 0 0 %g", zs[1]),
            string.format("H 0 0 %g", zs[2]),
            string.format("H 0 0 %g", zs[3]),
            string.format("H 0 0 %g", zs[4]),
            string.format("H 0 0 %g", zs[5]),
            string.format("H 0 0 %g", zs[6]),
        }
        end,
}

excit = {orders = {1, 2, 3}}

prop = {
    dt = 5e-7,
    propagator = "direct-overlap"
}

qmc = {
    initial_population = 1e3,
    target_population = 3e5,
    shift_damping = 1e-3,
    sampling_cutoff1 = 1e0,
    sampling_cutoff2 = 0.25,
    spawn_cutoff = 0.25,
    ncycles = 1e1,
    nreports = 562962,
    excitation_gen = "uniform",
}

states = {
    mom = {
        {label = "RHF (0, 0, 0, 0, 0, 0)", noci = true},
        {label = "UHF (1, -1, 1, -1, 1, -1)", spin_bias = {pattern = {1, -1, 1, -1, 1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1 , 1, -1, 1, -1, 1)", spin_bias = {pattern = {-1, 1, -1, 1, -1, 1}, pol = 0.75}, noci = true},
    },
}

write = {
    read_restart = "restarts/QMC_BENCHMARK3.3_RESTART.H5",
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}
