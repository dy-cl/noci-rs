<p align="center">
  <img src="assets/logo.png" alt="noci-rs logo" width="200">
</p>

<p align="center">
  <em>Nonorthogonal electronic structure in Rust.</em>
</p>

<p align="center">
  <a href="https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml"><img src="https://github.com/dy-cl/noci-rs/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/dy-cl/noci-rs/actions/workflows/clippy.yml"><img src="https://github.com/dy-cl/noci-rs/actions/workflows/clippy.yml/badge.svg?branch=main" alt="Clippy"></a>
  <a href="https://codecov.io/gh/dy-cl/noci-rs"><img src="https://codecov.io/gh/dy-cl/noci-rs/branch/main/graph/badge.svg" alt="Code coverage"></a>
  <a href="https://noci-rs.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/noci-rs/badge/?version=latest" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://github.com/dy-cl/noci-rs/actions/workflows/benchmarks.yml"><img src="https://dy-cl.github.io/noci-rs/badges/machines.svg" alt="Benchmark times by runner CPU"></a>
</p>

<h3 align="center">
  <a href="https://noci-rs.readthedocs.io/en/latest/getting-started/installation.html">Install</a>
  &nbsp;·&nbsp;
  <a href="https://noci-rs.readthedocs.io/en/latest/getting-started/quickstart.html">Quick start</a>
  &nbsp;·&nbsp;
  <a href="https://noci-rs.readthedocs.io/en/latest/methods/index.html">Methods</a>
  &nbsp;·&nbsp;
  <a href="https://noci-rs.readthedocs.io/en/latest/input/index.html">Input reference</a>
  &nbsp;·&nbsp;
  <a href="https://noci-rs.readthedocs.io/en/latest/citing.html">Citing</a>
</h3>

---

<p align="center">
  <img
    src="assets/results/HChain_STO-3G_1_5_Compression.png"
    width="820"
  >
</p>

<div align="center">

Shoulder population $N_w$ against hydrogen chain length $N_H$ for NOCISD(3)QMC and accuracy matched CIQMC in the STO-3G basis. The measured shoulders are extrapolated, demonstrating the large compression of NOCIQMC relative to CIQMC at comparable accuracy.

</div>

<p align="center">
  <img
    src="assets/results/H20_STO-3G_b-apply_NOCISD3_1_5.png"
    width="820"
  >
</p>

<div align="center">

Typical evolution of the projected energy $E_{\mathrm{Proj}}(\tau)$ and shift $E_s^S(\tau)$ for the H<sub>20</sub> linear chain at 1.5 Å in the STO-3G basis using the `b-apply` propagator. The same trajectories can also be seen for `s-apply`.

</div>

<p align="center">
  <img
    src="assets/results/F2_cc-pVDZ_NOCIPT2_Holomorphic_Scan.png"
    width="820"
  >
</p>

<div align="center">

NOCI-PT2(3) calculation for F<sub>2</sub> in the cc-pVDZ basis. The NOCI-PT2 method recovers much of the dynamical correlation absent from reference NOCI, shown by good agreement with the CCSD(T) energies. Use of the non-orthogonal Wick's theorem for matrix elements allows larger first-order spaces to be treated than was previously possible.

</div>

---

<p align="center">
  <sub>Research software under active development · <a href="LICENSE">MIT licence</a> · <a href="CONTRIBUTING.md">Contributing</a></sub>
</p>
