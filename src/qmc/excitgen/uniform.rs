// qmc/excitgen/uniform.rs

// External crate imports.
use rand::Rng;

// Crate-root imports.
use crate::determinant::OrthogonalConnection;
use crate::elements::MOCache;

/// Persistent uniform proposal topology for parent-orthogonal Hamiltonian connections.
pub(crate) struct OrthogonalUniformGenerator {
    /// Exact probability `1/N_\mathrm{conn}` for every table entry.
    pgen: f64,
    /// Complete system-wide table of relative occupied/virtual-rank connections.
    connections: Vec<OrthogonalConnection>,
}

impl OrthogonalUniformGenerator {
    /// Construct all `N_\mathrm{conn}` one- and two-body connection topologies once.
    /// The table enumerates `O_\alpha V_\alpha`, `O_\beta V_\beta`, same-spin pair products,
    /// and `O_\alpha V_\alpha O_\beta V_\beta` using orbital ranks rather than labels.
    /// # Arguments:
    /// - `occupations`: Representative alpha and beta occupations defining fixed electron counts.
    /// - `cache`: Representative MO cache defining common alpha and beta orbital dimensions.
    /// # Returns
    /// - `Self`: System-wide flat uniform connection table and valid-orbital masks.
    pub(crate) fn new(
        occupations: (u128, u128),
        cache: &MOCache<f64>,
    ) -> Self {
        // Derive occupied/virtual dimensions shared by every parent-orthogonal determinant.
        let noa = occupations.0.count_ones() as usize;
        let nob = occupations.1.count_ones() as usize;
        let nva = cache.ha.nrows() - noa;
        let nvb = cache.hb.nrows() - nob;
        // Count each Slater-Condon connection class and reserve the exact combined capacity.
        let nas = noa * nva;
        let nbs = nob * nvb;
        let naa = (noa * noa.saturating_sub(1) / 2) * (nva * nva.saturating_sub(1) / 2);
        let nbb = (nob * nob.saturating_sub(1) / 2) * (nvb * nvb.saturating_sub(1) / 2);
        let nab = nas * nbs;
        let mut connections = Vec::with_capacity(nas + nbs + naa + nbb + nab);

        // Enumerate alpha and beta single excitations in occupied/virtual rank coordinates.
        for occupied in 0..noa {
            for virtual_ in 0..nva {
                connections.push(OrthogonalConnection::AlphaSingle {
                    occupied: occupied as u8,
                    virtual_: virtual_ as u8,
                });
            }
        }
        for occupied in 0..nob {
            for virtual_ in 0..nvb {
                connections.push(OrthogonalConnection::BetaSingle {
                    occupied: occupied as u8,
                    virtual_: virtual_ as u8,
                });
            }
        }
        // Enumerate unique same-spin doubles with ordered occupied and virtual pairs.
        for occupied_i in 0..noa {
            for occupied_j in occupied_i + 1..noa {
                for virtual_a in 0..nva {
                    for virtual_b in virtual_a + 1..nva {
                        connections.push(OrthogonalConnection::AlphaDouble {
                            occupied: [occupied_i as u8, occupied_j as u8],
                            virtual_: [virtual_a as u8, virtual_b as u8],
                        });
                    }
                }
            }
        }
        for occupied_i in 0..nob {
            for occupied_j in occupied_i + 1..nob {
                for virtual_a in 0..nvb {
                    for virtual_b in virtual_a + 1..nvb {
                        connections.push(OrthogonalConnection::BetaDouble {
                            occupied: [occupied_i as u8, occupied_j as u8],
                            virtual_: [virtual_a as u8, virtual_b as u8],
                        });
                    }
                }
            }
        }
        // Enumerate all Cartesian products of alpha and beta single excitations.
        for occupied_a in 0..noa {
            for virtual_a in 0..nva {
                for occupied_b in 0..nob {
                    for virtual_b in 0..nvb {
                        connections.push(OrthogonalConnection::AlphaBetaDouble {
                            occupied_a: occupied_a as u8,
                            virtual_a: virtual_a as u8,
                            occupied_b: occupied_b as u8,
                            virtual_b: virtual_b as u8,
                        });
                    }
                }
            }
        }

        // Every retained topology has the same exact generation probability.
        let pgen = if connections.is_empty() {
            0.0
        } else {
            1.0 / connections.len() as f64
        };

        Self { pgen, connections }
    }

    /// Sample one connection with `P_\mathrm{gen}(D|x)=1/N_\mathrm{conn}`.
    /// # Arguments:
    /// - `self`: Persistent uniform connection topology.
    /// - `rng`: Thread-local random-number generator.
    /// # Returns
    /// - `Option<(OrthogonalConnection, f64)>`: Source-relative connection and exact uniform
    ///   probability.
    #[inline(always)]
    pub(crate) fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Option<(OrthogonalConnection, f64)> {
        if self.connections.is_empty() {
            None
        } else {
            Some((
                self.connections[rng.gen_range(0..self.connections.len())],
                self.pgen,
            ))
        }
    }
}
