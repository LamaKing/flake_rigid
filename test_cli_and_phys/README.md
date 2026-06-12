# CLI and physics integration tests

End-to-end acceptance tests for the `drift` CLI on a commensurate triangular
system.  These complement the unit tests in `tests/` by exercising the full
pipeline: YAML input → CLI → HDF5 output → physics checks.

## Requirements

Install the package first:

    cd ..
    pip install -e ".[dev]"

## Running

Execute from **this directory**:

    cd test_cli_and_phys
    bash run_tests.sh

Then check the physics output:

    python analyze.py

The script stops on the first error (`set -e`).  To rerun from scratch,
remove previous output first:

    rm -rf sweep_Fx_out sweep_tau_out map_trasl.h5 map_roto.h5 mep_roto.h5 mep_rototrasl.h5

## What is tested

| Command | Config | Output |
|---------|--------|--------|
| `drift map` | `grid_trasl.yaml` | `map_trasl.h5` |
| `drift map` | `grid_roto.yaml`  | `map_roto.h5`  |
| `drift string` | `string_roto.yaml` | `mep_roto.h5` |
| `drift string` | `string_rototrasl.yaml` | `mep_rototrasl.h5` |
| `drift sweep` | `sweep_Fx.yaml` | `sweep_Fx_out/` |
| `drift sweep` | `sweep_tau.yaml` | `sweep_tau_out/` |

`analyze.py` checks physical results: barrier heights, depinning thresholds,
and commensurate scaling.
