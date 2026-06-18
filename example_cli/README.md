# example_cli — CLI and physics examples

End-to-end examples for the `flake` CLI on a commensurate triangular system.
These complement the unit tests in `tests/` by exercising the full pipeline:
YAML input → CLI → HDF5 output → physics checks.

## Requirements

Install the package first:

    cd ..
    pip install -e ".[dev]"

## CLI acceptance test

Run all CLI commands and immediately check their physical output:

    cd example_cli
    bash run_tests.sh

`run_tests.sh` calls `flake map`, `flake string`, and `flake sweep` in
sequence, then runs `python analyze.py`.  The script stops on the first error
(`set -e`).

To rerun from scratch, remove previous output first:

    rm -rf sweep_Fx_out sweep_tau_out map_trasl.h5 map_roto.h5 mep_roto.h5 mep_rototrasl.h5

### What `run_tests.sh` covers

| Command | Config | Output |
|---------|--------|--------|
| `flake map` | `grid_trasl.yaml` | `map_trasl.h5` |
| `flake map` | `grid_roto.yaml`  | `map_roto.h5`  |
| `flake string` | `string_roto.yaml` | `mep_roto.h5` |
| `flake string` | `string_rototrasl.yaml` | `mep_rototrasl.h5` |
| `flake sweep` | `sweep_Fx.yaml` | `sweep_Fx_out/` |
| `flake sweep` | `sweep_tau.yaml` | `sweep_tau_out/` |

`analyze.py` checks physical results: barrier heights, depinning thresholds,
and commensurate scaling.

## Superlubricity demo

`superlubricity.py` is a standalone Python script (no CLI) that compares
translational depinning for a commensurate vs. an incommensurate cluster.
It demonstrates the friction-reduction effect directly through the Python API:

    python superlubricity.py

Tune `N1`, `N2`, `THETA_INC`, `EPSILON`, and `N_STEPS` at the top of the file
to explore different regimes.  Output: `superlubricity.png`.


