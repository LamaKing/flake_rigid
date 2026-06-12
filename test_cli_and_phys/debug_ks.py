#!/usr/bin/env python3
"""
Debug script 4: check exact array properties of ks in both paths,
and test if making ks explicitly contiguous fixes the YAML path.

Run from the test_cli directory:
    python3 debug_ks.py
"""

import numpy as np
import yaml

with open('params.yaml') as fh:
    params_yaml = yaml.safe_load(fh)

from drift.tool_create_substrate import substrate_from_params, get_ks

ks_from_yaml  = np.asarray(params_yaml['ks'], dtype=np.float64)
ks_from_get   = get_ks(1, 3, 4./3., 0.)
ks_from_yaml_c = np.ascontiguousarray(ks_from_yaml)

print('=== ks_from_yaml ===')
print('  shape:',    ks_from_yaml.shape)
print('  dtype:',    ks_from_yaml.dtype)
print('  C_CONTIGUOUS:', ks_from_yaml.flags['C_CONTIGUOUS'])
print('  F_CONTIGUOUS:', ks_from_yaml.flags['F_CONTIGUOUS'])
print('  values:\n', ks_from_yaml)

print('\n=== ks_from_get_ks ===')
print('  shape:',    ks_from_get.shape)
print('  dtype:',    ks_from_get.dtype)
print('  C_CONTIGUOUS:', ks_from_get.flags['C_CONTIGUOUS'])
print('  F_CONTIGUOUS:', ks_from_get.flags['F_CONTIGUOUS'])
print('  values:\n', ks_from_get)

print('\nValues identical:', np.allclose(ks_from_yaml, ks_from_get))

# --- Test: force ks to be contiguous in the YAML path ---
params_forced = dict(params_yaml)
params_forced['ks'] = np.ascontiguousarray(
    np.asarray(params_yaml['ks'], dtype=np.float64)
)

pen_yaml,   _, enp_yaml   = substrate_from_params(params_yaml)
pen_forced, _, enp_forced = substrate_from_params(params_forced)
pen_get,    _, enp_get    = substrate_from_params(
    {**params_yaml, 'ks': ks_from_get})

# Single point test
p0 = np.array([[0.3, 0.1]])
cm = np.zeros(2)

eA = pen_yaml  (p0, cm, *enp_yaml  )[0][0]
eB = pen_forced(p0, cm, *enp_forced)[0][0]
eC = pen_get   (p0, cm, *enp_get   )[0][0]

print('\nSingle point E at (0.3, 0.1):')
print('  YAML path:           %.8g' % eA)
print('  YAML + contiguous:   %.8g' % eB)
print('  get_ks path:         %.8g' % eC)

if abs(eB - eC) < abs(eA - eC):
    print('\n=> Making ks contiguous FIXES the bug.')
else:
    print('\n=> Contiguous flag is NOT the issue. Bug is elsewhere.')

# --- Quick visual confirmation ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

xx = np.linspace(-1.5, 1.5, 60)
yy = np.linspace(-1.5, 1.5, 60)
XX, YY = np.meshgrid(xx, yy)
p_grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

enA = pen_yaml  (p_grid, np.zeros(2), *enp_yaml  )[0]
enB = pen_forced(p_grid, np.zeros(2), *enp_forced)[0]
enC = pen_get   (p_grid, np.zeros(2), *enp_get   )[0]

fig, axes = plt.subplots(1, 3, dpi=120, figsize=(12, 4))
for ax, en, title in [
    (axes[0], enA, 'YAML ks'),
    (axes[1], enB, 'YAML ks + ascontiguousarray'),
    (axes[2], enC, 'get_ks'),
]:
    sc = ax.scatter(p_grid[:,0], p_grid[:,1], c=en, s=2, cmap='viridis')
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title('%s\nbarrier=%.3g' % (title, en.max()-en.min()))
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('debug_ks.png', dpi=120)
print('\nSaved debug_ks.png')
