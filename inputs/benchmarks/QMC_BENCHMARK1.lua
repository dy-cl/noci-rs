mol = {
    basis = 'cc-pVDZ',
    r = {2.8},
    unit = 'Ang',
    atoms = function(r)
        return {
            string.format("Li 0 0 %g", -r / 2), 
            string.format("H 0 0 %g",  r / 2),
        }
        end,
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = false},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
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
    ncycles = 1e1,
    nreports = 5e3,
    shift_damping = 5e-4,
    sampling_cutoff = 1e0,
    spawn_cutoff = 1e0,
    excitation_gen = 'uniform',
    seed = 1e0,
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}

