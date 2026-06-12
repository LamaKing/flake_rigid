#!/usr/bin/env python3
"""
Debug script 2: check what N we actually have, whether the cluster is
truly commensurate, and what the single-particle vs cluster landscape looks like.

Run from the test_cli directory:
    python3 debug_map2.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('params.yaml') as fh:
    params = yaml.safe_load(fh)

from flake.substrate import substrate_from_params
from flake.cluster   import cluster_from_params

pen_func, calc_en_f, en_params = substrate_from_params(params)
pos = cluster_from_params(params)
N = pos.shape[0]
print('N=%i' % N)
print('pos[0:3]:', pos[:3])
print('a1=%s  a2=%s' % (params['a1'], params['a2']))
print('ks[0]:', params['ks'][0])

# Single-particle energy at origin: should be -1 (epsilon=1)
e1, f1, t1 = pen_func(np.array([[0.,0.]]), np.zeros(2), *en_params)
print('\nSingle particle at origin: E=%.6g (expected -1.0)' % e1[0])

# Cluster at origin: should be -N
e0, f0, tau0 = calc_en_f(pos, np.zeros(2), *en_params)
print('Cluster at origin: E=%.6g (expected -%i)' % (e0, N))
print('|F| at origin: %.4g (expected ~0)' % np.linalg.norm(f0))

# Single-particle landscape: should always show hexagonal pattern
xx = np.linspace(-1.5, 1.5, 80)
yy = np.linspace(-1.5, 1.5, 80)
XX, YY = np.meshgrid(xx, yy)
p_grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

en_single, _, _ = pen_func(p_grid, np.zeros(2), *en_params)

# Cluster landscape
pos_cm_grid = np.array([[x, y] for x in xx for y in yy])
from flake.maps import translational_map
res = translational_map(pos, calc_en_f, en_params, None, 80, 80,
                        pos_cm_grid=pos_cm_grid, n_jobs=1)
en_cluster = res['energy']

print('\nSingle-particle barrier: %.4g' % (en_single.max()-en_single.min()))
print('Cluster barrier:         %.4g  (expected ~%.1f)' % (
    en_cluster.max()-en_cluster.min(), N*(en_single.max()-en_single.min())))

# Plot: single particle vs cluster
fig, axes = plt.subplots(1, 2, dpi=150, figsize=(9, 4))

sc0 = axes[0].scatter(p_grid[:,0], p_grid[:,1], c=en_single,
                      s=2, cmap='viridis', rasterized=True)
plt.colorbar(sc0, ax=axes[0], label='E', shrink=0.8)
axes[0].set_title('Single particle  barrier=%.3g' % (en_single.max()-en_single.min()))
axes[0].set_aspect('equal')

pp = res['pos_cm']
sc1 = axes[1].scatter(pp[:,0], pp[:,1], c=en_cluster,
                      s=2, cmap='viridis', rasterized=True)
plt.colorbar(sc1, ax=axes[1], label='E', shrink=0.8)
axes[1].set_title('Cluster N=%i  barrier=%.3g' % (N, en_cluster.max()-en_cluster.min()))
axes[1].set_aspect('equal')

for ax in axes:
    ax.set_xlabel('x_cm')
    ax.set_ylabel('y_cm')

plt.tight_layout()
plt.savefig('debug_map2.png', dpi=150)
print('\nSaved debug_map2.png')
plt.show()
