#!/bin/bash
# Run all DRIFT CLI tests for the commensurate triangular system.
# Execute from the directory containing params.yaml and the yaml config files.
#
# NOTE: sweep_md skips runs whose run_NNNN/params.yaml already exists (resume).
# To re-run from scratch, remove the output directories first:
#   rm -rf sweep_Fx_out sweep_tau_out

set -e   # stop on first error

echo "=== python ../src/cli.py map: translational ==="
python ../src/cli.py map -i params.yaml --grid grid_trasl.yaml -o map_trasl.h5

echo "=== python ../src/cli.py map: rotational ==="
python ../src/cli.py map -i params.yaml --grid grid_roto.yaml -o map_roto.h5

echo "=== python ../src/cli.py string: pure rotation ==="
python ../src/cli.py string -i params.yaml --cfg string_roto.yaml -o mep_roto.h5

echo "=== python ../src/cli.py string: roto-translational ==="
python ../src/cli.py string -i params.yaml --cfg string_rototrasl.yaml -o mep_rototrasl.h5

echo "=== python ../src/cli.py sweep: translational depinning (Fx) ==="
python ../src/cli.py sweep -i params.yaml --spec sweep_Fx.yaml

echo "=== python ../src/cli.py sweep: rotational depinning (Tau) ==="
python ../src/cli.py sweep -i params.yaml --spec sweep_tau.yaml

echo "=== All runs done. Run: python analyze.py ==="
python ./analyze.py


