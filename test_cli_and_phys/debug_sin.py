#!/usr/bin/env python3
"""
Debug script 3: isolate the sin substrate bug.

Compares:
  A) substrate_from_params loaded from YAML (CLI path)
  B) substrate_from_params built directly with get_ks (notebook path)

Run from the test_cli directory:
    python3 debug_sin.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('params.yaml') as fh:
    params_yaml = yaml.safe_load(fh)

from drift.tool_create_substrate import substrate_from_params, get_ks

# --- Path A: from YAML (what CLI does) ---
pen_A, en_A, enp_A = substrate_from_params(params_yaml)

# --- Path B: build ks with get_ks directly (what notebook does) ---
params_nb = dict(params_yaml)
params_nb['ks'] = get_ks(1, 3, 4./3., 0.)   # ndarray, not list-of-lists
pen_B, en_B, enp_B = substrate_from_params(params_nb)

# --- Compare ks seen by each path ---
ks_yaml = np.array(params_yaml['ks'])
ks_nb   = get_ks(1, 3, 4./3., 0.)
print('ks from YAML (type=%s):' % type(params_yaml['ks']))
print(ks_yaml)
print('\nks from get_ks (type=%s):' % type(ks_nb))
print(ks_nb)
print('\nMax diff:', np.abs(ks_yaml - ks_nb).max())

# --- Single particle on a grid ---
xx = np.linspace(-1.5, 1.5, 80)
yy = np.linspace(-1.5, 1.5, 80)
XX, YY = np.meshgrid(xx, yy)
p_grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

enA, _, _ = pen_A(p_grid, np.zeros(2), *enp_A)
enB, _, _ = pen_B(p_grid, np.zeros(2), *enp_B)

print('\nPath A (YAML ks):  E range [%.4g, %.4g]' % (enA.min(), enA.max()))
print('Path B (get_ks):   E range [%.4g, %.4g]' % (enB.min(), enB.max()))
print('Max diff between A and B:', np.abs(enA - enB).max())

fig, axes = plt.subplots(1, 2, dpi=150, figsize=(9, 4))
for ax, en, title in [(axes[0], enA, 'YAML ks (CLI path)'),
                      (axes[1], enB, 'get_ks (notebook path)')]:
    sc = ax.scatter(p_grid[:,0], p_grid[:,1], c=en, s=2,
                    cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=ax, label='E', shrink=0.8)
    ax.set_title('%s\nbarrier=%.3g' % (title, en.max()-en.min()))
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('debug_sin.png', dpi=150)
print('\nSaved debug_sin.png')
plt.show()
