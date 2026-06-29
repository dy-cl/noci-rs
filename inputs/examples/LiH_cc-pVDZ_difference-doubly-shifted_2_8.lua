mol = {
    basis = "cc-pVDZ",
    r = {2.8},
    unit = "Ang",
    atoms = function(r)
        return {
            string.format("Li 0 0 %g", -r / 2),
            string.format("H 0 0 %g",   r / 2),
        }
    end,
}

scf = {
    max_cycle = 1e5,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

states = {
    mom = {
        {
            label = "RHF (0, 0)",
            noci = false,
        },
        {
            label = "UHF (1, -1)",
            spin_bias = {
                pol = 0.75,
                pattern = {1, -1},
            },
            noci = true,
        },
        {
            label = "UHF (-1, 1)",
            spin_bias = {
                pol = 0.75,
                pattern = {-1, 1},
            },
            noci = true,
        },
    },
}

excit = {
    orders = {1, 2},
}

prop = {
    dt = 1e-4,
    propagator = "difference-doubly-shifted-u2",
}

qmc = {
    initial_population = 5e2,
    target_population = 1.5e4,
    shift_damping = 5e-4,
    ncycles = 1e1,
    nreports = 5e5,
    excitation_gen = "uniform",
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}
