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
    expansion_rank = 4,

    dt = 1e-2,

    qmc = {
        excitation_gen = "uniform",

        initial_population = 1000,
        target_population = 1200,

        shift_damping = 1e-2,

        ncycles = 5,
        nreports = 24,

        fri = {
            population = {
                cutoff = 1e-1,
            },

            spawn = {
                cutoff = 1e-2,
            },
        },

        seed = 7,
    },
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
