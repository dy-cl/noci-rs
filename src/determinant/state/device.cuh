// determinant/state/device.cuh
// `DeviceDeterminant`, laid out as the struct of the same name in `device.rs`, with the bitstring
// operations the device kernels apply to its occupations, counterparts of those of `Bits` in
// `determinant/bits.rs`.

#pragma once

/// @brief One determinant: its occupations and parent, with its excitations relative to that
/// parent.
struct DeviceDeterminant {
    /// Alpha and beta occupation bitstrings, orbital `p` at bit `p % 64` of word `p / 64`.
    unsigned long long occupation[2][WORDS];
    /// Parent orbital frame.
    unsigned int parent;
    /// Excitation rank of the alpha and beta spin strings.
    unsigned char rank[2];
    /// Hole orbitals of the alpha and beta spin strings, ascending.
    unsigned char holes[2][MAXEXCIT];
    /// Particle orbitals of the alpha and beta spin strings, ascending.
    unsigned char particles[2][MAXEXCIT];
    /// Product of the alpha and beta excitation phases.
    double phase;
};

/// @brief Number of set orbitals below `p`, the counterpart of `Bits::count_below`.
/// @param bits Bitstring of `WORDS` words.
/// @param p Orbital.
/// @return The number of set orbitals `q < p`.
__device__ int count_below(const unsigned long long *bits, int p) {
    int full = p >> 6;
    int count = 0;
    for (int k = 0; k < full; ++k) {
        count += __popcll(bits[k]);
    }
    if ((p & 63) != 0) {
        count += __popcll(bits[full] & ((1ULL << (p & 63)) - 1));
    }
    return count;
}

/// @brief Set orbitals in increasing order, the counterpart of `Bits::ones`.
/// @param bits Bitstring of `WORDS` words.
/// @param out Orbitals to write, at most `max`.
/// @param max Largest number of orbitals to write.
/// @return The number of set orbitals, which may exceed `max`.
__device__ int ones(const unsigned long long *bits, int *out, int max) {
    int count = 0;
    for (int k = 0; k < WORDS; ++k) {
        unsigned long long word = bits[k];
        while (word != 0) {
            if (count < max) {
                out[count] = 64 * k + __ffsll(word) - 1;
            }
            ++count;
            word &= word - 1;
        }
    }
    return count;
}

/// @brief Fermionic phase of the excitation `a^\dagger_{parts} a_{holes}` of `occupation`, the
/// counterpart of `excitation_phase_bits`: annihilators act from the highest hole down, then
/// creators from the lowest particle up.
/// @param occupation Source bitstring of `WORDS` words.
/// @param holes Hole orbitals, ascending.
/// @param nhole Number of holes.
/// @param particles Particle orbitals, ascending.
/// @param nparticle Number of particles.
/// @return `+1` or `-1`.
__device__ double excitation_phase(
    const unsigned long long *occupation,
    const int *holes,
    int nhole,
    const int *particles,
    int nparticle
) {
    unsigned long long occ[WORDS];
    for (int k = 0; k < WORDS; ++k) {
        occ[k] = occupation[k];
    }

    int odd = 0;
    for (int h = nhole - 1; h >= 0; --h) {
        int i = holes[h];
        odd ^= count_below(occ, i) & 1;
        occ[i >> 6] &= ~(1ULL << (i & 63));
    }
    for (int k = 0; k < nparticle; ++k) {
        int a = particles[k];
        odd ^= count_below(occ, a) & 1;
        occ[a >> 6] |= 1ULL << (a & 63);
    }

    return odd ? -1.0 : 1.0;
}
