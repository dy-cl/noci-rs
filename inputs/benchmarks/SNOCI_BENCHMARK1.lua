mol = {
    basis = 'cc-pVDZ',
    r = {2.8},
    unit = 'Ang',
    atoms = function(r)
        return {string.format("Li 0 0 %g", -r / 2), string.format("H 0 0 %g",  r / 2),}
        end,
}

excit = {orders = {1, 2}}

snoci = {
    max_iter = 1,
    gmres = {
        max_iter = 512,
        restart = 256,
        res_tol = 1e-6,
        full_m = true,
    },
}

states = {
    mom = {
        {label = "RHF (0, 0)", noci = false},
        {label = "UHF (1, -1)", spin_bias = {pattern = {1, -1}, pol = 0.75}, noci = true},
        {label = "UHF (-1, 1)", spin_bias = {pattern = {-1, 1}, pol = 0.75}, noci = true},
    }
}

wicks = {
    enabled = true, 
    compare = false,
    storage = "ram",
    cachedir = ".",
}
