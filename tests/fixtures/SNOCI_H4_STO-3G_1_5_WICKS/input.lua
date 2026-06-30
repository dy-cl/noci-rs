scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = "STO-3G",
    r = {1.5},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("H 0 0 %g", -1.5 * r),
            string.format("H 0 0 %g", -0.5 * r),
            string.format("H 0 0 %g",  0.5 * r),
            string.format("H 0 0 %g",  1.5 * r),
        }
    end,
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = true},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1, 1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1, -1, 1}, pol = 0.75}, noci = true},
    }
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}

excit = {
    orders = {2},
}

snoci = {
    max_iter = 2,
    max_add = 1,

    gmres = {
        max_iter = 512,
        restart = 64,
        res_tol = 1e-10,
        metric_tol = 1e-10,
        full_m = true,
    },
}
