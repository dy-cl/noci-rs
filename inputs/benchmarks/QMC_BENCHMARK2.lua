mol = {
    basis = 'cc-pVDZ',
    r = {1.5},
    unit = 'Ang',
    atoms = function(r)
        local zs = {-1.5 * r, -0.5 * r, 0.5 * r, 1.5 * r}
        return {
            string.format("H 0 0 %g", zs[1]),
            string.format("H 0 0 %g", zs[2]),
            string.format("H 0 0 %g", zs[3]),
            string.format("H 0 0 %g", zs[4]),
        }
        end,
}

states = {
    mom = {
        {label = "RHF (0, 0, 0, 0)", noci = true},
        {label = "UHF (1, -1, 1, -1)", spin_bias = {pattern = {1, -1, 1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1 , 1, -1, 1)", spin_bias = {pattern = {-1, 1, -1, 1}, pol = 0.75}, noci = true},
    },
}

excit = {orders = {1, 2}}

prop = {
    dt = 1e-4,
    max_steps = 5e4,
    propagator = "difference-doubly-shifted-u2",
}

qmc = {
    initial_population = 1e3,
    target_population = 1e5,
    ncycles = 10,
    nreports = 5000,
    shift_damping = 0.0005,
    excitation_gen = 'uniform',
    seed = 1,
}

wicks = {
    enabled = true,
    compare = true,
    storage = "ram",
    cachedir = ".",
}

