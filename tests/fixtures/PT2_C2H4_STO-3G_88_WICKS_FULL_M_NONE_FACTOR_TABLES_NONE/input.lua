scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
}

mol = {
    basis = "STO-3G",
    r = {88.0},
    unit = "Ang",
    atoms = function(r)
        local rcc = 1.3206136
        local rch = 1.0839274533
        local hcc = math.rad(121.66453473)
        local phi = math.rad(r)
        local xh = rch * math.cos(hcc)
        local yh = rch * math.sin(hcc)
        return {
            string.format("C %g 0 0", -0.5 * rcc),
            string.format("C %g 0 0",  0.5 * rcc),
            string.format("H %g %g 0", -0.5 * rcc + xh,  yh),
            string.format("H %g %g 0", -0.5 * rcc + xh, -yh),
            string.format("H %g %g %g", 0.5 * rcc - xh,  yh * math.cos(phi),  yh * math.sin(phi)),
            string.format("H %g %g %g", 0.5 * rcc - xh, -yh * math.cos(phi), -yh * math.sin(phi)),
        }
    end,
}

states = {
    mom = {
        {
            label = "RHF",
            noci = true,
        },
        {
            label = "RHF (pi*)^2",
            excit = {spin = "both", occ = -1, vir = 0},
            noci = true,
        },
        {
            label = "UHF (1, -1)",
            spin_bias = {
                pattern = {1, -1, 0, 0, 0, 0},
                pol = 0.10,
            },
            noci = true,
        },
        {
            label = "UHF (-1, 1)",
            spin_bias = {
                pattern = {-1, 1, 0, 0, 0, 0},
                pol = 0.10,
            },
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

    full_m = "none",
    factor_tables = "none",
    gmres = {
        max_iter = 1024,
        restart = 1024,
        res_tol = 1e-5,
    },
}
