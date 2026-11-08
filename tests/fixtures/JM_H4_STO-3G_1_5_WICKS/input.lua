scf = {
    max_cycle = 1e5,
    e_tol = 1e-12,
    fds_sdf_tol = 1e-6,
    d_tol = 1e-4,
    diis = {
        space = 8,
    },
}

mol = {
    basis = 'STO-3G',
    r = {1.5},
    unit = 'Ang',
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
        {
            label = "UHF (+-+-)",
            spin_bias = {
                pattern = {1, -1, 1, -1},
                pol = 0.75,
            },
            noci = true,
        },
        {
            label = "UHF (-+-+)",
            spin_bias = {
                pattern = {-1, 1, -1, 1},
                pol = 0.75,
            },
            noci = true,
        },
    },
}

jmnoccmc = {
    cluster_rank = 2,
    dt = 0.1,
    max_steps = 1e5,
    residual_tol = 1e-10,
    e_tol = 1e-12,
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
