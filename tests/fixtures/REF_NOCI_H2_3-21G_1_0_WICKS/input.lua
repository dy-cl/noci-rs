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
    r = {1.40, 1.38, 1.36, 1.34, 1.32, 1.30, 1.28, 1.26, 1.24, 1.22, 1.20, 1.18, 1.16, 1.14, 1.12, 1.10, 1.08, 1.06, 1.04, 1.02, 1.00},
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
    enabled = true,
    compare = true,
    storage = "RAM",
    cachedir = ".",
}
