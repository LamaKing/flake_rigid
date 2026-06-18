"""
Tests for sweep_md.py.

Unit tests (no marker): run with plain `pytest`.
Slow tests (marker `slow`): run with `pytest -m slow`.
    These include the parallel-vs-serial benchmark, which uses the same
    cluster geometry as the depinning notebook (N=85, triangular substrate,
    epsilon=1) but with n_steps=2000 and 6 force values so it finishes
    in a few seconds on any modern laptop.

To run only the benchmark:
    pytest tests/test_sweep_md.py -m slow -v
"""

import os
import sys
import warnings
import tempfile

import numpy as np
from numpy import sqrt
import pytest

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import cluster_from_params, calc_cluster_langevin
from flake.sweep import (
    grid_sweep, line_sweep, force_sweep, concat_sweeps,
    last_state, mean_velocity, drift_velocity,
    sweep_md, load_sweep,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_KS = get_ks(1.0, 3, 4.0 / 3.0, 0.0)

_PARAMS = {
    'sub_basis':     [[0, 0]],
    'epsilon':       1.0,
    'well_shape':    'sin',
    'ks':            _KS,
    'a1':            np.array([1.0, 0.0]),
    'a2':            np.array([0.5, -sqrt(3.0) / 2.0]),
    'cl_basis':      [[0, 0]],
    'cluster_shape': 'circle',
    'N1': 9, 'N2': 9,
    'theta': 0.0, 'pos_cm': [0, 0],
}


@pytest.fixture(scope='module')
def substrate():
    _, en_func, _ = substrate_from_params(_PARAMS)
    return en_func


@pytest.fixture(scope='module')
def cluster():
    return cluster_from_params(_PARAMS)


# ---------------------------------------------------------------------------
# 1-4. Grid helpers (pure Python, no MD)
# ---------------------------------------------------------------------------

def test_grid_sweep_shape():
    """Cartesian product of 3 x 2 = 6 points."""
    spec = grid_sweep({'Fx': [0.0, 0.1, 0.2], 'kBT': [0.0, 1.0]})
    assert len(spec) == 6
    assert spec[0]  == {'Fx': 0.0, 'kBT': 0.0}
    assert spec[-1] == {'Fx': 0.2, 'kBT': 1.0}


def test_line_sweep_basic():
    spec = line_sweep({'Tau': [0, 1, 2], 'Fx': [0, 0, 0]})
    assert len(spec) == 3
    assert spec[1] == {'Tau': 1, 'Fx': 0}


def test_line_sweep_length_mismatch():
    with pytest.raises(ValueError, match='length'):
        line_sweep({'Tau': [0, 1], 'Fx': [0]})


def test_force_sweep_decomposition():
    spec = force_sweep([0.0, 1.0], phi_deg=90.0)
    assert len(spec) == 2
    assert abs(spec[1]['Fx'])        < 1e-12
    assert abs(spec[1]['Fy'] - 1.0) < 1e-12


def test_concat_sweeps_dedup():
    s = force_sweep([0.0, 0.5], phi_deg=0.0)
    cat = concat_sweeps(s, s)
    assert len(cat) == 2  # duplicates dropped


def test_invalid_key_raises():
    with pytest.raises(ValueError, match='bad_key'):
        sweep_md(np.array([[0., 0.]]),
                 lambda *a: (0., np.zeros(2), 0.),
                 [{'bad_key': 1.0}],
                 base_md_kwargs={'eta': 1.0})


# ---------------------------------------------------------------------------
# 5-7. post_fn helpers
# ---------------------------------------------------------------------------

def _make_traj(n=20):
    """Synthetic trajectory dict for post_fn unit tests."""
    t      = np.linspace(0., 1., n)
    pos_cm = np.column_stack([t * 0.5, np.zeros(n)])
    vel_cm = np.column_stack([np.full(n, 0.5), np.zeros(n)])
    return {'t': t, 'pos_cm': pos_cm, 'vel_cm': vel_cm,
            'theta': np.zeros(n), 'energy': -np.ones(n),
            'omega': np.zeros(n), 'force': np.zeros((n, 2)),
            'torque': np.zeros(n)}


def test_last_state_keys():
    traj = _make_traj()
    out  = last_state()(traj, {})
    assert set(out.keys()) == {'pos_cm', 'theta', 'energy', 'vel_cm', 'omega'}
    assert np.allclose(out['pos_cm'], traj['pos_cm'][-1])


def test_mean_velocity_value():
    traj = _make_traj()
    # All vel_cm rows are [0.5, 0], so mean speed = 0.5.
    v = mean_velocity(1.0)(traj, {})
    assert abs(v - 0.5) < 1e-12


def test_drift_velocity_value():
    traj = _make_traj()
    # pos_cm goes from [0,0] to [0.5, 0] over t in [0,1].
    vd = drift_velocity()(traj, {})
    # x-drift = 0.5/1.0 = 0.5, y-drift = 0.
    assert vd.shape == (2,)
    assert abs(vd[0] - 0.5) < 1e-12
    assert abs(vd[1])        < 1e-12


# ---------------------------------------------------------------------------
# 8. sweep_md serial smoke test
# ---------------------------------------------------------------------------

def test_sweep_md_serial_returns_correct_structure(substrate, cluster):
    en_func = substrate
    spec = grid_sweep({'Fx': [0.0, 0.1]})
    base = {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
            'n_steps': 200, 'print_every': 50}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = sweep_md(cluster, en_func, spec,
                           base_md_kwargs=base, save=False, verbose=False)
    assert len(results) == 2
    assert results[0]['run_dir'] is None
    assert 'energy' in results[0]['result']
    assert abs(results[0]['params']['Fx'] - 0.0) < 1e-12
    assert abs(results[1]['params']['Fx'] - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# 9. save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(substrate, cluster):
    en_func = substrate
    spec = grid_sweep({'Fx': [0.0, 0.1]})
    base = {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
            'n_steps': 200, 'print_every': 100}
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sweep_md(cluster, en_func, spec,
                     base_md_kwargs=base, save=True, outdir=tmp, verbose=False)

        dirs = sorted(os.listdir(tmp))
        assert len(dirs) == 2
        assert dirs[0].startswith('run_0000')
        assert dirs[1].startswith('run_0001')

        loaded = load_sweep(tmp)
        assert len(loaded) == 2
        assert 'energy' in loaded[0]['result']
        assert abs(loaded[0]['params']['Fx'] - 0.0) < 1e-12
        assert abs(loaded[1]['params']['Fx'] - 0.1) < 1e-12


# ---------------------------------------------------------------------------
# 10. Parallel benchmark  (pytest -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_loky_faster_than_explicit_loop(substrate):
    """loky n_jobs=4 must be faster than a serial run_md loop.

    Uses a larger cluster (N~145, tmp.py benchmark) and 8 force values so
    each worker gets enough work to amortise loky process-startup cost.
    n_steps=200000, dt=5e-4 per run; total wall time < 2 min on 4 cores.

    The serial baseline calls run_md directly (JIT path, no callbacks),
    which is the fastest possible serial path.  Speedup > 1 is the only
    assertion -- the exact value depends on core count and machine load.

    Run with:  pytest tests/test_sweep_md.py -m slow -v -s
    """
    from time import time
    from flake.dynamics import run_md as _run_md
    from flake.cluster import make_cluster

    en_func = substrate

    A1  = np.array([1.0, 0.0])
    A2  = np.array([0.5, -sqrt(3.0) / 2.0])
    pos = make_cluster(A1, A2, 15, 15, shape='circle')
    N   = len(pos)

    eta      = 1.0
    kBT      = 1e-5
    dt       = 5e-4
    n_steps  = 200000
    pos_cm0  = np.array([0.0, 0.0])
    F_values = [0., 100., 250., 255., 260., 265., 300., 500.]

    # warmup: compile JIT for this cluster size
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _run_md(pos, en_func, eta=eta, kBT=kBT, dt=dt,
                n_steps=10, print_every=10, seed=0)

    # serial baseline: plain run_md loop (JIT path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time()
        for Fx in F_values:
            _run_md(pos, en_func,
                    eta=eta, Fx=Fx, kBT=kBT, dt=dt, n_steps=n_steps,
                    theta0=0.0, pos_cm0=pos_cm0.copy(), print_every=n_steps)
        t_loop = time() - t0

    # parallel: sweep_md with loky n_jobs=4
    spec = [{'Fx': float(f)} for f in F_values]
    base = {'eta': eta, 'kBT': kBT, 'dt': dt, 'n_steps': n_steps,
            'print_every': n_steps, 'pos_cm0': pos_cm0.copy()}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time()
        sweep_md(pos, en_func, spec,
                 base_md_kwargs=base, post_fn=drift_velocity(),
                 n_jobs=4, backend='loky', save=False, verbose=False)
        t_loky = time() - t0

    speedup = t_loop / max(t_loky, 1e-6)
    print("\n--- parallel benchmark  N=%d  n_runs=%d ---" % (N, len(F_values)))
    print("serial loop  : %.2f s" % t_loop)
    print("loky n_jobs=4: %.2f s" % t_loky)
    print("speedup      : %.2fx" % speedup)

    assert speedup > 1.0, (
        "loky n_jobs=4 (%.1fs) not faster than serial (%.1fs); speedup %.2fx"
        % (t_loky, t_loop, speedup)
    )
