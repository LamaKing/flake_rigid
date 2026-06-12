#!/usr/bin/env python3
"""
Debug script: replicate exactly what 'drift map' does for a translational
map with well_shape='sin', and compare with a direct notebook-style call.

Run from the test_cli directory:
    python3 debug_map.py

Expected output: both maps should show the same hexagonal pattern and
the same barrier value.  If they differ, the bug is in the CLI path.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------
# Load physics -- same as _build_physics in cli.py
# ---------------------------------------------------------------------------

with open('params.yaml') as fh:
    params = yaml.safe_load(fh)

print(params)
from drift.tool_create_substrate import substrate_from_params
from drift.tool_create_cluster   import cluster_from_params

_, calc_en_f, en_params = substrate_from_params(params)
pos = cluster_from_params(params)
print('N=%i  en_params=%s' % (pos.shape[0], en_params))

# ---------------------------------------------------------------------------
# Load grid config -- same as _cmd_map in cli.py
# ---------------------------------------------------------------------------

with open('grid_trasl.yaml') as fh:
    grid = yaml.safe_load(fh)

print('grid.yaml contents:', grid)

n_x        = int(grid.get('n_x', 50))
n_y        = int(grid.get('n_y', 50))
well_shape = params.get('well_shape', 'sin')

print('well_shape=%s  n_x=%i  n_y=%i' % (well_shape, n_x, n_y))

# Replicate CLI grid construction for sin substrate
if well_shape == 'sin':
    if 'x_range' not in grid or 'y_range' not in grid:
        print('ERROR: x_range/y_range missing from grid.yaml')
        raise SystemExit(1)
    x_range = list(grid['x_range'])
    y_range = list(grid['y_range'])
    print('x_range=%s  y_range=%s' % (x_range, y_range))

    xx = np.linspace(x_range[0], x_range[1], n_x)
    yy = np.linspace(y_range[0], y_range[1], n_y)
    pos_cm_grid_cli = np.array([[x, y] for x in xx for y in yy])
    print('pos_cm_grid shape (CLI):', pos_cm_grid_cli.shape)
    print('pos_cm_grid first 3 rows:', pos_cm_grid_cli[:3])
    print('pos_cm_grid last  3 rows:', pos_cm_grid_cli[-3:])

# ---------------------------------------------------------------------------
# Notebook-style grid (what works in the notebook)
# ---------------------------------------------------------------------------

xx_nb = np.linspace(-1.5, 1.5, n_x)
yy_nb = np.linspace(-1.5, 1.5, n_y)
# Notebook uses meshgrid then ravel -- different ordering from CLI!
XX, YY       = np.meshgrid(xx_nb, yy_nb)
pos_cm_grid_nb = np.stack([XX.ravel(), YY.ravel()], axis=1)
print('\npos_cm_grid shape (notebook):', pos_cm_grid_nb.shape)
print('pos_cm_grid first 3 rows (notebook):', pos_cm_grid_nb[:3])

# ---------------------------------------------------------------------------
# Are the two grids the same set of points?
# ---------------------------------------------------------------------------

# Sort both by (x, y) to compare regardless of ordering
cli_sorted = pos_cm_grid_cli[np.lexsort(pos_cm_grid_cli[:, ::-1].T)]
nb_sorted  = pos_cm_grid_nb [np.lexsort(pos_cm_grid_nb [:, ::-1].T)]

if np.allclose(cli_sorted, nb_sorted, atol=1e-10):
    print('\nGrids are IDENTICAL (same points, possibly different order)')
else:
    print('\nGrids DIFFER -- this is the bug')
    print('CLI  x range: [%.4g, %.4g]' % (pos_cm_grid_cli[:,0].min(),
                                            pos_cm_grid_cli[:,0].max()))
    print('NB   x range: [%.4g, %.4g]' % (pos_cm_grid_nb[:,0].min(),
                                            pos_cm_grid_nb[:,0].max()))

# ---------------------------------------------------------------------------
# Compute maps with both grids
# ---------------------------------------------------------------------------

from drift.maps import translational_map

print('\nComputing CLI-style map ...')
res_cli = translational_map(pos, calc_en_f, en_params, None, n_x, n_y,
                             pos_cm_grid=pos_cm_grid_cli, n_jobs=1)

print('Computing notebook-style map ...')
res_nb  = translational_map(pos, calc_en_f, en_params, None, n_x, n_y,
                             pos_cm_grid=pos_cm_grid_nb, n_jobs=1)

print('\nCLI    barrier: %.4g' % (res_cli['energy'].max() - res_cli['energy'].min()))
print('NB     barrier: %.4g' % (res_nb ['energy'].max() - res_nb ['energy'].min()))

# ---------------------------------------------------------------------------
# Plot both side by side
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, dpi=150, figsize=(9, 4))

for ax, res, title in [(axes[0], res_cli, 'CLI grid'),
                       (axes[1], res_nb,  'Notebook grid')]:
    pp = res['pos_cm']
    en = res['energy']
    sc = ax.scatter(pp[:, 0], pp[:, 1], c=en, s=2, cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=ax, label='E', shrink=0.8)
    ax.set_title('%s  barrier=%.3g' % (title, en.max() - en.min()))
    ax.set_xlabel('x_cm')
    ax.set_ylabel('y_cm')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('debug_map.png', dpi=150)
print('\nSaved debug_map.png')
plt.show()
