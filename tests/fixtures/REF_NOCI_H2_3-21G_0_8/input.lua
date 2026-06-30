scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = "3-21G",
    r = {2.0, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8},
    unit = "Ang",
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
        {
            label = "UHF (+, -)",
            spin_bias = {
                pattern = {1, -1},
                pol = 0.75,
            },
            noci = true,
        },
        {
            label = "UHF (-, +)",
            spin_bias = {
                pattern = {-1, 1},
                pol = 0.75,
            },
            noci = true,
        },
        {
            label = "h-UHF (+, -)",
            holomorphic = true,
            partner = "UHF (+, -)",
            spin_bias = {
                pattern = {1, -1},
                pol = 0.75,
            },
            noci = true,
        },
        {
            label = "h-UHF (-, +)",
            holomorphic = true,
            partner = "UHF (-, +)",
            spin_bias = {
                pattern = {-1, 1},
                pol = 0.75,
            },
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
