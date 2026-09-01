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
    orders = {1, 2},
}

snoci = {
    max_iter = 1,

    gmres = {
        max_iter = 1024,
        restart = 1024,
        res_tol = 1e-10,
        metric_tol = 1e-10,
        full_m = "none",
        factor_tables = "none",
    },
}
