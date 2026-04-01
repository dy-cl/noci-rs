scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = 'STO-3G',
    r = {1.75},
    unit = 'Ang',
    atoms = function(r)
        return {string.format("F 0 0 %g", -r / 2), string.format("F 0 0 %g",  r / 2),}
        end,
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = true},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

prop = {
    dt = 1e-3,
    max_steps = 5e6,
    propagator = "unshifted",
}

det = {
    dynamic_shift = true,
    dynamic_shift_alpha = 0.1,
    e_tol = 1e-12,
}

wicks = {
    enabled = false,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
