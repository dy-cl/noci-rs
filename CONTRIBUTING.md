# Contributing

Contributions to `noci-rs` are welcome. For substantial changes, please open an
issue first so the approach can be discussed before significant work begins.

The [code architecture guide](docs/development/architecture.md) maps the source
modules, shared interfaces and dependency boundaries for contributors.

## Development workflow

1. Fork the repository and create a branch from `main`.
2. Make a change, following the conventions in [STYLE.md](STYLE.md).
3. Add or update tests and documentation where appropriate.
4. Run the checks below.
5. Push the branch to your fork and open a pull request against `main`.

The build requirements and setup are described in the
[installation guide](https://noci-rs.readthedocs.io/en/latest/getting-started/installation.html).

## Checks

Before opening a pull request, run:

```bash
cargo fmt --all --check
cargo clippy --locked --all-targets --no-default-features -- -D warnings
./scripts/test.sh cargo test --locked --release -p noci-rs --no-default-features
```

Some numerical tests are sensitive to BLAS and CPU differences. If a test fails
only because of a small numerical discrepancy, include the failure and details
of your platform in the pull request.

Do not use `--features gnocc` or `--all-features` unless your change concerns the
experimental NOCC implementation, as compiling these can take a large
amount of time.

## Documentation

The user documentation is a Sphinx site under [`docs/`](docs/), written in MyST
Markdown and published on [Read the Docs](https://noci-rs.readthedocs.io). Keep it
in step with the code in the same pull request:

- **New or changed input options**: update the table's page under `docs/input/`,
  with the key's type, default as the parser actually resolves it, and any
  validation or propagator restrictions.
- **New methods or changed behaviour**: update the relevant page under
  `docs/methods/`, and `docs/guide/advice.md` if it changes how calculations
  should be set up.
- **New literature references**: add a BibTeX entry to `docs/references.bib` and
  cite it with `` {cite}`key` ``. Code comments keep their own references as
  described in [STYLE.md](STYLE.md).
- **New result figures**: add the image under `assets/results/` and show it on
  `docs/results.md`.

The README is only a landing page with the logo, badges and links into the
documentation; put explanations and results in the docs instead.

Build the documentation locally with warnings treated as errors, as Read the
Docs does:

```bash
python3 -m venv .docsenv
.docsenv/bin/pip install -r docs/requirements.txt
.docsenv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

## Commit messages

Use the Conventional Commits format:

```text
<type>(<scope>): <description>
```

The supported types are:

- `feat`: Add or change user-facing functionality.
- `fix`: Correct a bug.
- `perf`: Improve performance without changing behaviour.
- `refactor`: Restructure code without fixing a bug or adding functionality.
- `test`: Add or update tests.
- `docs`: Change documentation only.
- `build`: Change dependencies, build scripts, or packaging.
- `ci`: Change continuous-integration configuration.
- `chore`: Perform repository maintenance not covered above.

Every commit must include a short scope identifying the affected area, for
example `scf`, `qmc`, `wicks`, `input`, or `tests`. Use `repo` for changes that
apply to the repository as a whole.

```text
feat(qmc): add overlap-weighted excitation sampling
fix(scf): preserve the MOM state ordering
perf(wicks): cache repeated excitation phases
test(pt2): cover singular overlap matrices
docs(readme): clarify MPI configuration
ci(tests): pin the OpenBLAS version
```

- Write the description in the imperative mood, in lower case, without a full stop.
- Keep each commit focused on one logical change.
- Use the body to explain why the change is needed or to note important
  implementation details.

## Pull requests

Use the following template when opening a pull request:

````markdown
## Summary

<!-- What changed, and why? -->

## Testing

<!-- List the checks you ran. Include relevant platform details for numerical failures. -->

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --locked --all-targets --no-default-features -- -D warnings`
- [ ] `./scripts/test.sh cargo test --locked --release -p noci-rs --no-default-features`
- [ ] `sphinx-build -W --keep-going -b html docs docs/_build/html`, if `docs/` changed

## Checklist

- [ ] The change is focused and follows `STYLE.md`.
- [ ] Tests were added or updated where appropriate.
- [ ] Documentation under `docs/` was added or updated where appropriate.
- [ ] Commit messages follow the prescribed style.
````
