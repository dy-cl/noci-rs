# noci-rs

:::{div} hero
Nonorthogonal electronic structure in Rust.
:::

`noci-rs` performs electronic-structure calculations in determinant spaces built from
independently optimised, and therefore mutually nonorthogonal, orbital sets. It finds multiple
real and holomorphic SCF solutions, couples them by reference nonorthogonal configuration
interaction (NOCI), and recovers the remaining correlation by NOCI-PT2, selected NOCI or
stochastic imaginary-time propagation (NOCIQMC) in the excited nonorthogonal space.
Nonorthogonal coupled cluster (NOCC and JM-NOCCMC) is under development.

Matrix elements use the Slater–Condon rules within one orbital frame, and the generalised
Slater–Condon rules or the extended nonorthogonal Wick's theorem between frames
{cite}`burton2021,burton2022`. Calculations run in parallel through Rayon within a node and MPI
across nodes.

:::{important}
`noci-rs` is research software under active development. Input formats, numerical interfaces
and experimental methods may change.
:::

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/installation
:link-type: doc

Build the code, set up basis-set data and run a first reference NOCI calculation.
:::

:::{grid-item-card} {octicon}`book` Methods
:link: methods/index
:link-type: doc

The theory behind each method: SCF states, NOCI, NOCI-PT2, NOCIQMC and NOCC.
:::

:::{grid-item-card} {octicon}`list-unordered` Input reference
:link: input/index
:link-type: doc

Every table and option of the Lua input, with defaults and constraints.
:::

:::{grid-item-card} {octicon}`light-bulb` Practical advice
:link: guide/advice
:link-type: doc

Choosing propagators, time steps, populations and generators, and analysing the output.
:::

:::{grid-item-card} {octicon}`graph` Results
:link: results
:link-type: doc

Example calculations: NOCIQMC compression, a stochastic trajectory and a NOCI-PT2 scan.
:::

:::{grid-item-card} {octicon}`mortar-board` Citing
:link: citing
:link-type: doc

How to cite `noci-rs` and the methods it implements.
:::
::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/running
```

```{toctree}
:hidden:
:caption: Methods

methods/index
methods/scf
methods/noci
methods/elements
methods/selected
methods/nociqmc
methods/nocc
```

```{toctree}
:hidden:
:caption: Input reference

input/index
input/molecule
input/scf
input/states
input/excit
input/propagation
input/snoci
input/reshf
input/write
input/wicks
input/nocc
```

```{toctree}
:hidden:
:caption: Guide

guide/advice
results
```

```{toctree}
:hidden:
:caption: About

citing
references
Contributing <https://github.com/dy-cl/noci-rs/blob/main/CONTRIBUTING.md>
```
