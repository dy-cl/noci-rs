scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    fds_sdf_tol = 1e-8,
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
            string.format("H 0 0 %g", -1.5 * r),
            string.format("H 0 0 %g", -0.5 * r),
            string.format("H 0 0 %g", 0.5 * r),
            string.format("H 0 0 %g", 1.5 * r),
        }
    end,
}

states = {
    mom = {
        { label = "RHF", noci = true },
    },
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

reshf = {
    weights = {1.0},
    cutoff = 1e-8,
    space = "excitations",
}
