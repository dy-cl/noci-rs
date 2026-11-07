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
        return {string.format("H 0 0 %g", -r / 2), string.format("H 0 0 %g",  r / 2),}
        end,
}

states = {
    mom = {
        {
            label = "RHF (0, 0)",
            noci = true,
        },
    },
}

wicks = {
    enabled = false,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}

reshf = {
    weights = {1.0},
    cutoff = 1e-8,
}
