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

wicks = {
    enabled = false,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
