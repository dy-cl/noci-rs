scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = 'STO-3G',
    r = {1.75},
    unit = 'Ang',
    atoms = function(a)
        local h = a / 2
        return {
            string.format("H %g %g 0.0", -h, -h),
            string.format("H %g %g 0.0", -h,  h),
            string.format("H %g %g 0.0",  h, -h),
            string.format("H %g %g 0.0",  h,  h),
        }
    end,
}

states = {
    mom = {
        ---------------------------------------- RHF -----------------------------------------------------
        {label = "RHF (1, 1, -1, -1)", noci = true, spatial_bias = {pattern = {1, 1, -1, -1}, pol = 1e-2}},
        {label = "RHF (1, -1, 1, -1)", noci = true, spatial_bias = {pattern = {1, -1, 1, -1}, pol = 1e-2}},
        ---------------------------------------- UHF -----------------------------------------------------
        --- No zeros. These are the four-fold degenerate states.
        {label = "UHF (1, 1, -1, -1)", noci = true, spin_bias = {pattern = {1, 1, -1, -1}, pol = 0.75}},
        {label = "UHF (-1, -1, 1, 1)", noci = true, spin_bias = {pattern = {-1, -1, 1, 1}, pol = 0.75}},
    },
}

wicks = {
    enabled = true,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
