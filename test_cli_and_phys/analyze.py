#!/usr/bin/env python3
"""
Analysis script for DRIFT CLI test runs.

Loads HDF5 outputs produced by:
    drift map    -i params.yaml --grid grid_trasl.yaml  -o map_trasl.h5
    drift map    -i params.yaml --grid grid_roto.yaml   -o map_roto.h5
    drift string -i params.yaml --cfg  string_roto.yaml     -o mep_roto.h5
    drift string -i params.yaml --cfg  string_rototrasl.yaml -o mep_rototrasl.h5
    drift sweep  -i params.yaml --spec sweep_Fx.yaml
    drift sweep  -i params.yaml --spec sweep_tau.yaml

Checks:
  - Translational map: barrier = max(E) - min(E), symmetry of landscape.
  - Rotational map:    period = 60 deg, minima at multiples of 60 deg.
  - MEP roto:         barrier value, path stays near y=0.
  - MEP rototrasl:    barrier vs roto-only, coupled x-theta motion.
  - Depinning Fx:     Fc vs analytical prediction Fc = N * F1s ~ 237.4.
  - Depinning Tau:    tau_c visible as onset of omega > 0.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load(path):
    from drift.slides_io import load_map
    result, params = load_map(path)
    return result, params


def _load_sweep_dir(outdir):
    from drift.sweep_md import load_sweep, filter_sweep
    raw = load_sweep(outdir)
    return filter_sweep(raw)   # drops None-result entries with warning if > 10%


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------

def analytical_Fc(N):
    """Depinning force for N commensurate particles on triangular sin substrate.

    F1s = max |dV/dx| for a single particle at y=0.
    V = -eps/n^2 * |sum_l exp(i k_l r)|^2,  n=3, get_ks(1,3,4/3,0).

    This is an exact result for a commensurate cluster: all particles sit
    at equivalent positions, so each contributes F1s independently.
    Fc = N * F1s.
    """
    from numpy import pi
    R, c_n, n_k = 1.0, 4./3., 3
    ks = np.array([c_n * pi / R * np.array([np.cos(2.*pi/n_k*l),
                                             np.sin(2.*pi/n_k*l)])
                   for l in range(n_k)])
    inv_n2 = 1. / (n_k * n_k)
    x = np.linspace(0., 2.*pi / (c_n*pi/R), 100000)
    Fx = np.zeros_like(x)
    for xi in range(len(x)):
        sc = [(np.cos(ks[l,0]*x[xi]), np.sin(ks[l,0]*x[xi])) for l in range(n_k)]
        sum_cos = sum(c for c,s in sc)
        sum_sin = sum(s for c,s in sc)
        sum_kx_sin = sum(ks[l,0]*sc[l][1] for l in range(n_k))
        sum_kx_cos = sum(ks[l,0]*sc[l][0] for l in range(n_k))
        Fx[xi] = -1.0 * 2.*inv_n2 * (sum_cos*sum_kx_sin - sum_sin*sum_kx_cos)
    F1s = np.max(np.abs(Fx))
    return F1s, N * F1s


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _panel_scatter(ax, x, y, c, label, cmap='viridis', s=1):
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, rasterized=True)
    ax.set_title(label, fontsize=8)
    return sc


# ---------------------------------------------------------------------------
# 1. Translational map
# ---------------------------------------------------------------------------

def plot_trasl_map(h5path='map_trasl.h5'):
    if not os.path.exists(h5path):
        print('SKIP: %s not found' % h5path); return

    result, _ = _load(h5path)
    pp    = result['pos_cm']
    en    = result['energy']
    F     = result['force']
    tau   = result['torque']

    barrier = en.max() - en.min()
    print('Translational map: E_min=%.4g  E_max=%.4g  barrier=%.4g' %
          (en.min(), en.max(), barrier))

    # Reshape for contour (assumes square grid)
    n = int(np.sqrt(len(en)))
    if n*n == len(en):
        xx = pp[:,0].reshape(n,n)
        yy = pp[:,1].reshape(n,n)
        zz = en.reshape(n,n)

    fig, axes = plt.subplots(1, 4, dpi=150, figsize=(12, 2.8), sharey=True)
    axE, axFx, axFy, axTau = axes
    s0 = 2

    from matplotlib.colors import Normalize
    sc = axE.scatter(pp[:,0], pp[:,1], c=en, s=s0, cmap='viridis', rasterized=True)
    plt.colorbar(sc, ax=axE, label=r'$E$', shrink=0.8)

    for ax, vals, label, cmap in [
        (axFx,  F[:,0], r'$F_x$',  'PiYG'),
        (axFy,  F[:,1], r'$F_y$',  'PiYG'),
        (axTau, tau,    r'$\tau$', 'RdBu'),
    ]:
        vmax = np.abs(vals).max()
        sc2 = ax.scatter(pp[:,0], pp[:,1], c=vals, s=s0, cmap=cmap,
                         norm=Normalize(-vmax, vmax), rasterized=True)
        plt.colorbar(sc2, ax=ax, label=label, shrink=0.8)

    for ax in axes:
        ax.set_xlabel(r'$x_\mathrm{cm}$')
        ax.set_aspect('equal')
    axE.set_ylabel(r'$y_\mathrm{cm}$')
    axE.set_title('Translational map  barrier=%.4g' % barrier, fontsize=8)
    plt.tight_layout()
    plt.savefig('plot_trasl_map.png', dpi=150)
    plt.show()
    print('Saved: plot_trasl_map.png')


# ---------------------------------------------------------------------------
# 2. Rotational map
# ---------------------------------------------------------------------------

def plot_roto_map(h5path='map_roto.h5'):
    if not os.path.exists(h5path):
        print('SKIP: %s not found' % h5path); return

    result, _ = _load(h5path)
    theta = result['theta']
    en    = result['energy']
    tau   = result['torque']

    # Check period: should be 60 deg for triangular-on-triangular
    # Find minima positions
    from scipy.signal import argrelmin
    imin = argrelmin(en, order=5)[0]
    if len(imin) >= 2:
        periods = np.diff(theta[imin])
        print('Rotational map: detected minima at theta =', theta[imin])
        print('  periods between minima:', periods, '(expected: 60 deg)')
    else:
        print('Rotational map: fewer than 2 minima detected -- check range')

    barrier_r = en.max() - en.min()
    print('Rotational barrier = %.4g' % barrier_r)

    fig, (axE, axTau) = plt.subplots(2, 1, dpi=150, figsize=(6, 4), sharex=True)
    axE.plot(theta, en, '-k', lw=0.8)
    axE.set_ylabel('Energy')
    axE.set_title('Rotational map  barrier=%.4g' % barrier_r, fontsize=8)
    # Mark expected period lines
    for th in np.arange(0, theta.max()+1, 60):
        axE.axvline(th, ls=':', color='tab:blue', lw=0.7, alpha=0.6)

    axTau.plot(theta, tau, '-b', lw=0.8)
    axTau.axhline(0, ls=':', color='gray', lw=0.7)
    axTau.set_ylabel(r'Torque $\tau$')
    axTau.set_xlabel(r'$\theta$ (deg)')
    # Torque zero crossings should coincide with energy extrema
    for th in np.arange(0, theta.max()+1, 60):
        axTau.axvline(th, ls=':', color='tab:blue', lw=0.7, alpha=0.6)

    plt.tight_layout()
    plt.savefig('plot_roto_map.png', dpi=150)
    plt.show()
    print('Saved: plot_roto_map.png')


# ---------------------------------------------------------------------------
# 3 & 4. String MEP
# ---------------------------------------------------------------------------

def plot_mep(h5path, title, outname):
    if not os.path.exists(h5path):
        print('SKIP: %s not found' % h5path); return

    result, _ = _load(h5path)
    pts  = result['points']    # (n_pt, 2 or 3)
    en   = result['energy']
    grad = result['gradient']
    dim  = pts.shape[1]
    s    = np.linspace(0, 1, len(pts))

    barrier = en.max() - en.min()
    print('%s: dim=%d  barrier=%.5g' % (title, dim, barrier))
    if dim == 3:
        print('  max|y_cm| = %.3g (should be ~0 by symmetry)' % np.abs(pts[:,1]).max())

    if dim == 2:
        fig, (axP, axE) = plt.subplots(1, 2, dpi=150, figsize=(8, 3))
        axP.plot(pts[:,0], pts[:,1], '-k', lw=0.8)
        sc = axP.scatter(pts[:,0], pts[:,1], c=en, cmap='magma', s=8, zorder=3)
        plt.colorbar(sc, ax=axP, label='E')
        axP.set_xlabel(r'$x_\mathrm{cm}$'); axP.set_ylabel(r'$y_\mathrm{cm}$')
        axP.set_aspect('equal')

    else:  # dim == 3
        fig, axes = plt.subplots(1, 3, dpi=150, figsize=(11, 3))
        axP, axE, axG = axes
        # Path in (x, theta) space
        sc = axP.scatter(pts[:,0], pts[:,2], c=en, cmap='magma', s=8, zorder=3)
        axP.plot(pts[:,0], pts[:,2], '-k', lw=0.5, zorder=2)
        plt.colorbar(sc, ax=axP, label='E')
        axP.set_xlabel(r'$x_\mathrm{cm}$'); axP.set_ylabel(r'$\theta$ (deg)')
        axP.set_title('max|y|=%.2g' % np.abs(pts[:,1]).max(), fontsize=8)
        # Gradient components
        axG.plot(s, grad[:,0], '-b', label=r'$F_x$')
        axG.plot(s, grad[:,2], '-r', label=r'$\tau$')
        axG.axhline(0, ls=':', color='gray', lw=0.7)
        axG.set_xlabel('path'); axG.set_ylabel('gradient')
        axG.legend(fontsize=7)

    # Energy profile (always)
    ax_en = axE if dim == 2 else axE
    ax_en.plot(s, en, '-k')
    sc2 = ax_en.scatter(s, en, c=en, cmap='magma', s=15, zorder=2)
    ax_en.axhline(en.max(), ls='--', color='red', lw=0.8,
                  label='barrier=%.4g' % barrier)
    ax_en.set_xlabel('path (norm. arc length)')
    ax_en.set_ylabel('energy')
    ax_en.legend(fontsize=7)

    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.show()
    print('Saved:', outname)


# ---------------------------------------------------------------------------
# 5. Depinning sweep -- translational (Fx)
# ---------------------------------------------------------------------------

def plot_depinning_Fx(outdir='sweep_Fx_out', N=85):
    if not os.path.exists(outdir):
        print('SKIP: %s not found' % outdir); return

    data = _load_sweep_dir(outdir)
    if not data:
        print('ERROR: no complete runs in %s' % outdir); return
    Fx_vals = np.array([d['params']['Fx'] for d in data])

    # load_sweep returns full traj dicts; apply drift_velocity here.
    # With save_traj=False runs have result=None and are dropped by filter_sweep.
    def _vdrift(traj):
        pos_cm = traj['pos_cm']
        t      = traj['t']
        dt_tot = float(t[-1] - t[0])
        if dt_tot == 0.:
            return np.zeros(2)
        return (pos_cm[-1] - pos_cm[0]) / dt_tot

    vx = np.array([_vdrift(d['result'])[0] for d in data])

    F1s, Fc_analytical = analytical_Fc(N)
    print('Depinning Fx: F1s=%.4f  Fc_analytical=%.4f (N=%i)' % (F1s, Fc_analytical, N))

    # Estimate numerical Fc: midpoint between last pinned and first sliding
    thresh = 1e-3
    pinned  = vx < thresh
    sliding = ~pinned
    if pinned.any() and sliding.any():
        Fc_lo = Fx_vals[pinned].max()
        Fc_hi = Fx_vals[sliding].min()
        print('  Numerical Fc in [%.2f, %.2f]  (analytical: %.2f)'
              % (Fc_lo, Fc_hi, Fc_analytical))
        rel_err = abs(0.5*(Fc_lo+Fc_hi) - Fc_analytical) / Fc_analytical
        print('  Relative error: %.1f%%' % (rel_err*100))
    else:
        print('  WARNING: no clear transition detected -- adjust Fx range')

    fig, ax = plt.subplots(dpi=150, figsize=(5, 3.5))
    ax.scatter(Fx_vals[pinned],  vx[pinned],
               color='tab:blue', label='pinned', zorder=3)
    ax.scatter(Fx_vals[sliding], vx[sliding],
               color='tab:red', marker='^', label='sliding', zorder=3)
    if pinned.any() and sliding.any():
        ax.axvspan(Fc_lo, Fc_hi, alpha=0.15, color='orange',
                   label=r'$F_c \in [%.1f, %.1f]$' % (Fc_lo, Fc_hi))
    ax.axvline(Fc_analytical, ls='--', color='green', lw=1.2,
               label=r'$F_c^\mathrm{anal}=%.1f$' % Fc_analytical)
    ax.set_xlabel(r'$F_x$')
    ax.set_ylabel(r'$v_\mathrm{drift}$')
    ax.legend(fontsize=7)
    ax.set_title('Translational depinning  N=%i' % N, fontsize=9)
    plt.tight_layout()
    plt.savefig('plot_depinning_Fx.png', dpi=150)
    plt.show()
    print('Saved: plot_depinning_Fx.png')


# ---------------------------------------------------------------------------
# 6. Depinning sweep -- rotational (Tau)
# ---------------------------------------------------------------------------

def plot_depinning_tau(outdir='sweep_tau_out'):
    if not os.path.exists(outdir):
        print('SKIP: %s not found' % outdir); return

    data = _load_sweep_dir(outdir)
    if not data:
        print('ERROR: no complete runs in %s' % outdir); return
    tau_vals    = np.array([d['params']['Tau'] for d in data])
    # load_sweep returns full traj dicts; extract final theta and omega
    theta_final = np.array([d['result']['theta'][-1]  for d in data])
    omega_final = np.array([d['result']['omega'][-1]  for d in data])

    # Rotated if theta increased by more than one angular period (60 deg)
    # Use omega as the sliding indicator: omega > threshold means rotating
    thresh = 0.5   # deg/time; above this = depinned
    pinned  = np.abs(omega_final) < thresh
    sliding = ~pinned

    if pinned.any() and sliding.any():
        tc_lo = tau_vals[pinned].max()
        tc_hi = tau_vals[sliding].min()
        print('Depinning Tau: tau_c in [%.1f, %.1f]' % (tc_lo, tc_hi))
    else:
        tc_lo = tc_hi = None
        print('Depinning Tau: no clear transition -- adjust Tau range')

    fig, (axO, axTh) = plt.subplots(1, 2, dpi=150, figsize=(9, 3.5))

    axO.scatter(tau_vals[pinned],  omega_final[pinned],
                color='tab:blue', label='pinned', zorder=3)
    axO.scatter(tau_vals[sliding], omega_final[sliding],
                color='tab:red', marker='^', label='rotating', zorder=3)
    if tc_lo is not None:
        axO.axvspan(tc_lo, tc_hi, alpha=0.15, color='orange',
                    label=r'$\tau_c \in [%.0f, %.0f]$' % (tc_lo, tc_hi))
    axO.set_xlabel(r'$\tau_\mathrm{ext}$')
    axO.set_ylabel(r'$\omega_\mathrm{final}$ (deg/time)')
    axO.legend(fontsize=7)
    axO.set_title('Rotational depinning', fontsize=9)

    axTh.scatter(tau_vals[pinned],  theta_final[pinned],
                 color='tab:blue', label='pinned', zorder=3)
    axTh.scatter(tau_vals[sliding], theta_final[sliding],
                 color='tab:red', marker='^', label='rotating', zorder=3)
    axTh.axhline(60, ls='--', color='gray', lw=0.8, label='60 deg period')
    axTh.set_xlabel(r'$\tau_\mathrm{ext}$')
    axTh.set_ylabel(r'$\theta_\mathrm{final}$ (deg)')
    axTh.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('plot_depinning_tau.png', dpi=150)
    plt.show()
    print('Saved: plot_depinning_tau.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('DRIFT CLI test analysis')
    print('Run from the directory containing the output files')
    print('=' * 60)

    plot_trasl_map('map_trasl.h5')
    plot_roto_map('map_roto.h5')
    plot_mep('mep_roto.h5',     '3D string: pure rotation (0,0,0)->(0,0,60)',
             'plot_mep_roto.png')
    plot_mep('mep_rototrasl.h5','3D string: roto-trasl (0,0,0)->(1,0,60)',
             'plot_mep_rototrasl.png')
    plot_depinning_Fx('sweep_Fx_out',  N=85)
    plot_depinning_tau('sweep_tau_out')

    print('\nAll done.')
