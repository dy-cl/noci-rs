# Results

## NOCIQMC compression

```{figure} ../assets/results/HChain_STO-3G_1_5_Compression.png
:width: 100%
:alt: Shoulder population against hydrogen chain length for NOCISD(3)QMC and CIQMC

Shoulder population $N_w$ against hydrogen chain length $N_H$ for NOCISD(3)QMC and
accuracy-matched CIQMC in the STO-3G basis. The measured shoulders are extrapolated, showing the
large compression of NOCIQMC relative to CIQMC at comparable accuracy.
```

The shoulder is the population at which a stochastic calculation resolves the sign structure of
its wavefunction (see [the shoulder](methods/nociqmc.md#the-shoulder)), and so sets the population,
and therefore the memory and cost, that the calculation needs. A few nonorthogonal references
capture the static correlation of the stretched chain, leaving NOCIQMC to sample only the
remaining dynamic correlation.

## A stochastic trajectory

```{figure} ../assets/results/H20_STO-3G_b-apply_NOCISD3_1_5.png
:width: 100%
:alt: Projected energy and shift against iteration for H20 with b-apply

Typical evolution of the projected energy $E_\mathrm{Proj}(\tau)$ and shift $E_s^S(\tau)$ for the
H{sub}`20` linear chain at 1.5 Å in the STO-3G basis using the `b-apply` propagator. The same
trajectories are seen for `s-apply`.
```

The dotted line marks the start of population control. Before it the shift is fixed; after it the
shift fluctuates about the ground-state energy while the projected energy settles to a much less
noisy plateau, which `scripts/blocking.py` averages.

## NOCI-PT2 with holomorphic states

```{figure} ../assets/results/F2_cc-pVDZ_NOCIPT2_Holomorphic_Scan.png
:width: 100%
:alt: NOCI-PT2 binding curve of F2 in cc-pVDZ against CCSD(T)

NOCI-PT2(3) calculation for F{sub}`2` in the cc-pVDZ basis. NOCI-PT2 recovers much of the dynamic
correlation absent from reference NOCI, shown by good agreement with the CCSD(T) energies. Using
the nonorthogonal Wick's theorem for matrix elements allows larger first-order spaces to be treated
than was previously possible.
```

Holomorphic SCF states keep the NOCI basis continuous across the scan, so the NOCI-PT2 curve is
smooth where real solutions would coalesce and vanish. The input is in
[`inputs/examples/`](https://github.com/dy-cl/noci-rs/tree/main/inputs/examples).
