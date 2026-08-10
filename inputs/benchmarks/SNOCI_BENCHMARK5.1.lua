scf = {
    max_cycle = 1e5,
    e_tol = 1e-12,
    fds_sdf_tol = 1e-8,
    d_tol = 1e-6,
    do_fci = false,

    diis = {
        space = 15,
    },
}

mol = {
    basis = 'cc-pVDZ',
    r = {80.0},
    unit = 'Ang',

    atoms = function(theta)
        local rcc = 1.339
        local rch = 1.086
        local hcc = math.rad(121.2)
        local phi = math.rad(theta)

        local xc = rcc / 2.0
        local dx = -rch * math.cos(hcc)
        local rho = rch * math.sin(hcc)

        return {
            string.format("C %g 0 0", -xc),
            string.format("C %g 0 0",  xc),

            string.format("H %g %g 0", -xc - dx,  rho),
            string.format("H %g %g 0", -xc - dx, -rho),

            string.format(
                "H %g %g %g",
                xc + dx,
                rho * math.cos(phi),
                rho * math.sin(phi)
            ),
            string.format(
                "H %g %g %g",
                xc + dx,
                -rho * math.cos(phi),
                -rho * math.sin(phi)
            ),
        }
    end,
}

states = {
    mom = {
        {
            label = "RHF π^2",
            noci = true,
        },
        {
            label = "RHF (π*)^2",
            excit = {
                spin = "both",
                occ = -1,
                vir = 0,
            },
            noci = true,
        },
        {
            label = "RHF ionic (L)",
            spatial_bias = {
                pattern = {1, -1, 0, 0, 0, 0},
                pol = 0.10,
            },
            noci = true,
        },
        {
            label = "RHF ionic (R)",
            spatial_bias = {
                pattern = {-1, 1, 0, 0, 0, 0},
                pol = 0.10,
            },
            noci = true,
        },
        {
            label = "UHF diradical (L: A, R: B)",
            spin_bias = {
                pattern = {1, -1, 0, 0, 0, 0},
                pol = 0.10,
            },
            noci = true,
        },
        {
            label = "UHF diradical (L: B, R: A)",
            spin_bias = {
                pattern = {-1, 1, 0, 0, 0, 0},
                pol = 0.10,
            },
            noci = true,
        },
        {
            label = "UHF π->π* alpha",
            excit = {
                spin = "alpha",
                occ = -1,
                vir = 0,
            },
            noci = true,
        },
        {
            label = "UHF π->π* beta",
            excit = {
                spin = "beta",
                occ = -1,
                vir = 0,
            },
            noci = true,
        },
    },
}

excit = {
    orders = {1, 2},
}

snoci = {
    max_iter = 1,
    preconditioner = "woodbury",
    imag_shift = {
        0.0,
    },
    gmres = {
        max_iter = 512,
        restart = 128,
        res_tol = 1e-6,
        metric_tol = 1e-10,
        full_m = false,
    },
}

wicks = {
    enabled = true,
    compare = false,
    storage = "ram",
    cachedir = ".",
}

write = {
    verbose = 1,
    write_dir = "outputs/",
}
