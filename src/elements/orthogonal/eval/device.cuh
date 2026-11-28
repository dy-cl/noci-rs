// elements/orthogonal/eval/device.cuh
// Device counterparts of the orthogonal Slater-Condon evaluators in `fock.rs` and `hamiltonian.rs`
// with the connection of `connection_data` in `pairs.rs`, one same-parent determinant pair per
// thread in an orthonormal frame. They read the integrals that `DeviceOrthogonal` in
// `orthogonal/device.rs` copies. The bra is measured from the ket, so the ket occupations are the
// source of every sum over occupied orbitals.

#pragma once

#include "determinant/state/device.cuh"
#include "maths/wick/device.cuh"

/// @brief Offsets of one parent's integrals, laid out as `DeviceOrthogonalParent` in
/// `orthogonal/device.rs`.
struct DeviceOrthogonalParent {
    /// One when the Fock matrices are in an orthonormal frame and copied.
    unsigned long long fock;
    /// One when the Hamiltonian integrals are in an orthonormal frame and copied.
    unsigned long long hamiltonian;
    /// Number of molecular orbitals.
    unsigned long long n;
    /// Offset of the alpha Fock matrix, row-major `n x n`.
    unsigned long long fa;
    /// Offset of the beta Fock matrix, row-major `n x n`.
    unsigned long long fb;
    /// Offset of the alpha one-electron matrix, row-major `n x n`.
    unsigned long long ha;
    /// Offset of the beta one-electron matrix, row-major `n x n`.
    unsigned long long hb;
    /// Offset of the antisymmetrised alpha-alpha tensor `\langle pq||rs\rangle`, row-major `n^4`.
    unsigned long long aa;
    /// Offset of the antisymmetrised beta-beta tensor, row-major `n^4`.
    unsigned long long bb;
    /// Offset of the alpha-beta Coulomb tensor `(pq|rs)`, row-major `n^4`.
    unsigned long long ab;
};

/// @brief Excitation of the bra measured from the ket, the counterpart of `connection_data`.
/// @param bra Bra determinant.
/// @param ket Ket determinant of the same parent.
/// @param rank Largest total excitation rank the operator couples, at most two.
/// @param holes Hole orbitals of each spin to write, ascending.
/// @param particles Particle orbitals of each spin to write, ascending.
/// @param ranks Excitation rank of each spin to write.
/// @param phase Fermionic phase of the excitation to write.
/// @return The total rank, or `-1` when the operator cannot couple the pair.
__device__ int connection(
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    int rank,
    int holes[2][2],
    int particles[2][2],
    int ranks[2],
    double *phase
) {
    int total = 0;
    *phase = 1.0;
    for (int spin = 0; spin < 2; ++spin) {
        unsigned long long hole[WORDS];
        unsigned long long particle[WORDS];
        for (int k = 0; k < WORDS; ++k) {
            hole[k] = ket.occupation[spin][k] & ~bra.occupation[spin][k];
            particle[k] = bra.occupation[spin][k] & ~ket.occupation[spin][k];
        }

        int nhole = ones(hole, holes[spin], 2);
        int nparticle = ones(particle, particles[spin], 2);
        total += nhole;
        if (nhole != nparticle || total > rank) {
            return -1;
        }

        ranks[spin] = nhole;
        *phase *= excitation_phase(
            ket.occupation[spin], holes[spin], nhole, particles[spin], nparticle);
    }
    return total;
}

/// @brief Generalised-Fock and overlap elements of one same-parent pair, the counterpart of
/// `xw_fock_orthogonal_const`: the diagonal `\sum_{i\sigma} F^\sigma_{ii}` over the ket
/// occupations, or `p F^\sigma_{ai}` for a single.
/// @param parent Integrals of the shared parent.
/// @param integrals Device array of `DeviceOrthogonal`.
/// @param bra Bra determinant.
/// @param ket Ket determinant of the same parent.
/// @param f Generalised-Fock element to write.
/// @param s Overlap element to write.
__device__ void calculate_fs_pair_orthogonal(
    const DeviceOrthogonalParent &parent,
    const double *integrals,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double *f,
    double *s
) {
    int holes[2][2], particles[2][2], ranks[2];
    double phase;
    int total = connection(bra, ket, 1, holes, particles, ranks, &phase);
    *f = 0.0;
    *s = 0.0;
    if (total < 0) {
        return;
    }

    unsigned long long n = parent.n;
    const double *fock[2] = {integrals + parent.fa, integrals + parent.fb};

    if (total == 0) {
        // `F_xx = \sum_{i \in O_\alpha} F^\alpha_{ii} + \sum_{i \in O_\beta} F^\beta_{ii}`.
        double value = 0.0;
        for (int spin = 0; spin < 2; ++spin) {
            for (int k = 0; k < WORDS; ++k) {
                unsigned long long word = ket.occupation[spin][k];
                while (word != 0) {
                    int i = 64 * k + __ffsll(word) - 1;
                    value += fock[spin][i * n + i];
                    word &= word - 1;
                }
            }
        }
        *f = value;
        *s = 1.0;
        return;
    }

    // `F_{a \leftarrow i} = p F^\sigma_{ai}` for the excited spin.
    int spin = ranks[0] == 1 ? 0 : 1;
    *f = phase * fock[spin][particles[spin][0] * n + holes[spin][0]];
}

/// @brief Hamiltonian and overlap elements of one same-parent pair, the counterpart of
/// `xw_hamiltonian_orthogonal_const`.
/// @param parent Integrals of the shared parent.
/// @param integrals Device array of `DeviceOrthogonal`.
/// @param bra Bra determinant.
/// @param ket Ket determinant of the same parent.
/// @param enuc Nuclear repulsion energy.
/// @param h Hamiltonian element to write.
/// @param s Overlap element to write.
__device__ void calculate_hs_pair_orthogonal(
    const DeviceOrthogonalParent &parent,
    const double *integrals,
    const DeviceDeterminant &bra,
    const DeviceDeterminant &ket,
    double enuc,
    double *h,
    double *s
) {
    int holes[2][2], particles[2][2], ranks[2];
    double phase;
    int total = connection(bra, ket, 2, holes, particles, ranks, &phase);
    *h = 0.0;
    *s = 0.0;
    if (total < 0) {
        return;
    }

    unsigned long long n = parent.n;
    const double *one[2] = {integrals + parent.ha, integrals + parent.hb};
    const double *same[2] = {integrals + parent.aa, integrals + parent.bb};
    const double *ab = integrals + parent.ab;
    int occupied[2][64 * WORDS];
    int nocc[2] = {
        ones(ket.occupation[0], occupied[0], 64 * WORDS),
        ones(ket.occupation[1], occupied[1], 64 * WORDS),
    };

    if (total == 0) {
        // `H_xx = E_nuc + \sum_{i\sigma} h^\sigma_{ii} + \frac12 \sum_{ij\sigma} \langle ii||jj\rangle
        //        + \sum_{i \in O_\alpha, j \in O_\beta} (ii|jj)`.
        double value = enuc;
        for (int spin = 0; spin < 2; ++spin) {
            for (int k = 0; k < nocc[spin]; ++k) {
                int i = occupied[spin][k];
                value += one[spin][i * n + i];
            }
        }
        for (int spin = 0; spin < 2; ++spin) {
            for (int k = 0; k < nocc[spin]; ++k) {
                for (int l = 0; l < nocc[spin]; ++l) {
                    int i = occupied[spin][k];
                    int j = occupied[spin][l];
                    value += 0.5 * same[spin][tensor(n, i, i, j, j)];
                }
            }
        }
        for (int k = 0; k < nocc[0]; ++k) {
            for (int l = 0; l < nocc[1]; ++l) {
                int i = occupied[0][k];
                int j = occupied[1][l];
                value += ab[tensor(n, i, i, j, j)];
            }
        }
        *h = value;
        *s = 1.0;
        return;
    }

    if (total == 1) {
        // `H_{a \leftarrow i} = p [h^\sigma_{ai} + \sum_{j \in O_\sigma \setminus i} (ai||jj)
        //                          + \sum_{j \in O_{\bar\sigma}} (ai|jj)]`, with the alpha-beta
        // tensor indexed alpha first.
        int spin = ranks[0] == 1 ? 0 : 1;
        int other = 1 - spin;
        int i = holes[spin][0];
        int a = particles[spin][0];
        double value = one[spin][a * n + i];
        for (int k = 0; k < nocc[spin]; ++k) {
            int j = occupied[spin][k];
            if (j != i) {
                value += same[spin][tensor(n, a, i, j, j)];
            }
        }
        for (int k = 0; k < nocc[other]; ++k) {
            int j = occupied[other][k];
            value += spin == 0 ? ab[tensor(n, a, i, j, j)] : ab[tensor(n, j, j, i, a)];
        }
        *h = phase * value;
        return;
    }

    // Doubles: `p \langle ai||jb\rangle` within one spin, `p (ai|jb)` across the spins.
    if (ranks[0] == 2 || ranks[1] == 2) {
        int spin = ranks[0] == 2 ? 0 : 1;
        int i = holes[spin][0];
        int j = holes[spin][1];
        int a = particles[spin][0];
        int b = particles[spin][1];
        *h = phase * same[spin][tensor(n, a, i, j, b)];
    } else {
        *h = phase * ab[tensor(n, particles[0][0], holes[0][0], holes[1][0], particles[1][0])];
    }
}
