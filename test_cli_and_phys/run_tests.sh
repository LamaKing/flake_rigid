#!/bin/bash
# Run all FLAKE CLI tests for the commensurate triangular system.
# Execute from the directory containing params.yaml and the yaml config files.
# Requires the package to be installed: pip install -e ".[dev]"
#
# NOTE: sweep_md skips runs whose run_NNNN/params.yaml already exists (resume).
# To re-run from scratch, remove the output directories first:
#   rm -rf sweep_Fx_out sweep_tau_out

set -e   # stop on first error

echo "=== flake map: translational ==="
flake map -i params.yaml --grid grid_trasl.yaml -o map_trasl.h5

echo "=== flake map: rotational ==="
flake map -i params.yaml --grid grid_roto.yaml -o map_roto.h5

echo "=== flake string: pure rotation ==="
flake string -i params.yaml --cfg string_roto.yaml -o mep_roto.h5

echo "=== flake string: roto-translational ==="
flake string -i params.yaml --cfg string_rototrasl.yaml -o mep_rototrasl.h5

echo "=== flake sweep: translational depinning (Fx) ==="
flake sweep -i params.yaml --spec sweep_Fx.yaml

echo "=== flake sweep: rotational depinning (Tau) ==="
flake sweep -i params.yaml --spec sweep_tau.yaml

echo "=== All runs done. Run: python analyze.py ==="
python ./analyze.py
