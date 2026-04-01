scf = {
    max_cycle = 1e4,
    e_tol = 1e-12,
    diis = {
        space = 8,
    },
    do_fci = false,
}

mol = {
    basis = 'cc-pVDZ',
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
        {label = "UHF (1, -1, 1, -1)", noci = true, spin_bias = {pattern = {1, -1, 1, -1}, pol = 0.75}},
        {label = "UHF (-1, 1, -1, 1)", noci = true, spin_bias = {pattern = {-1, 1, -1, 1}, pol = 0.75}},
        -- Two zeros. There should be two lots of two-fold degenerate states.
        {label = "UHF (1, -1, 0, 0)", noci = true, spin_bias = {pattern = {1, -1, 0, 0}, pol = 0.75}}, 
        {label = "UHF (-1, 1, 0, 0)", noci = true, spin_bias = {pattern = {-1, 1, 0, 0}, pol = 0.75}},
        {label = "UHF (1, 0, 0, -1)", noci = true, spin_bias = {pattern = {1, 0, 0, -1}, pol = 0.75}}, 
        {label = "UHF (-1, 0, 0, 1)", noci = true, spin_bias = {pattern = {-1, 0, 0, 1}, pol = 0.75}},
    },
}

wicks = {
    enabled = false,
    compare = false,
    storage = "RAM",
    cachedir = ".",
}
