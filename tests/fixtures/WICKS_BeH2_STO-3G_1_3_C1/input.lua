scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
}

mol = {
    basis = 'STO-3G',
    r = {1.3},
    unit = 'Ang',
    atoms = function(r)
        return {
            "Be 0 0 0",
            string.format("H 0.3 0.2 %g", -r),
            string.format("H -0.1 0.4 %g", 0.95 * r),
        }
    end,
}

excit = {
    orders = {1, 2, 3},
}

states = {
    mom = {
        {
            label = "RHF",
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
