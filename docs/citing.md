# Citing noci-rs

`noci-rs` does not yet have a dedicated software publication. Until it does, please cite the
repository,

```bibtex
@software{noci-rs,
  author = {Dylan Counsell},
  title  = {noci-rs: Nonorthogonal electronic structure in Rust},
  url    = {https://github.com/dy-cl/noci-rs},
  year   = {2026},
}
```

together with the publications for the methods you use:

| Method | Cite |
| --- | --- |
| Reference NOCI | {cite}`thom2009,burton2019` |
| SCF metadynamics, MOM | {cite}`thom2008,gilbert2008` |
| Holomorphic SCF and tracking | {cite}`burton2016,burton2019complex` |
| Nonorthogonal Wick's theorem | {cite}`burton2021,burton2022` |
| NOCI-PT2 and selected NOCI | {cite}`burton2020` |
| ResHF | {cite}`miller2025` |
| NOCIQMC propagators | {cite}`baumgarten2023,booth2009` |
| Fast randomised iteration | {cite}`greene2019,lim2017,weare2026` |
| PCHB excitation generation | {cite}`holmes2016,weser2023` |
| GNOCC | {cite}`kutzelnigg1997,lee2026,lee2026symmetry` |
| JM-NOCCMC | {cite}`jeziorski1981,thom2010,franklin2016` |

Integrals are computed by libcint {cite}`sun2015` and basis sets are read from the Basis Set
Exchange {cite}`pritchard2019`.

`noci-rs` is distributed under the terms in
[`LICENSE`](https://github.com/dy-cl/noci-rs/blob/main/LICENSE).
