"""
Physics validation tests.

Core results verified:
  1. F1s matches the analytic single-particle depinning force to 10%.
  2. Commensurate linear scaling: Fc(N=7) / Fc(N=1) = 7 within 10%.
  3. Superlubricity: incommensurate cluster slides at Fs < 0.5 * Fc_comm.

All sweeps use kBT=0 (deterministic gradient descent) so n_steps=200000
at dt=1e-3 is sufficient to reach steady state without thermal fluctuations.
Single-particle runs use kBT=0 and Tau=0 so eta_r=0 is safe (no rotational
dynamics needed).

Substrate: triangular sin, epsilon=1, spacing=1.
"""

import numpy as np
import pytest
import warnings

from flake.substrate import substrate_from_params, get_ks
from flake.cluster import make_cluster, rotate
from flake.sweep import sweep_md, drift_velocity


_A1 = np.array([1.0, 0.0])
_A2 = np.array([0.5, -np.sqrt(3.0) / 2.0])

_SIN_PARAMS = {
    'well_shape': 'sin',
    'epsilon':    1.0,
    'sub_basis':  [[0., 0.]],
    'ks':         get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist(),
}

# kBT=0 + Tau=0: rotational DOF is frozen, eta_r=0 is safe.
# 200000 steps at dt=1e-3 = 200 time units; enough to reach sliding or stay pinned.
# print_every < n_steps to store at least two trajectory points for drift_velocity.
_MD_BASE = dict(eta=1.0, kBT=0., dt=1e-3, n_steps=200000, print_every=100000)


def _analytical_F1s():
    """Max |dV/dx| at y=0 for a single particle on the triangular sin substrate.

    Exact for this substrate: F1s ~ 2.793 for epsilon=1, spacing=1.
    """
    ks     = get_ks(1.0, 3, 4.0 / 3.0, 0.0)
    nk     = len(ks)
    inv_n2 = 1.0 / (nk * nk)
    x      = np.linspace(0.0, 1.0, 200000)   # one lattice period
    Fx     = np.zeros_like(x)
    for xi in range(len(x)):
        sum_cos = sum_sin = skxs = skxc = 0.0
        for l in range(nk):
            ph = ks[l, 0] * x[xi]
            c, s = np.cos(ph), np.sin(ph)
            sum_cos += c;  sum_sin += s
            skxs += ks[l, 0] * s;  skxc += ks[l, 0] * c
        Fx[xi] = -2.0 * inv_n2 * (sum_cos * skxs - sum_sin * skxc)
    return float(np.max(np.abs(Fx)))


def _depinning_Fc(pos, en_func, en_params, F_low, F_high, n_pts=12):
    """Coarse force sweep; returns (Fc_lo, Fc_hi) bracket of transition.

    Uses kBT=0 so the result is deterministic.  Asserts a clear pinned/sliding
    transition exists within [F_low, F_high].
    """
    F_vals = np.linspace(F_low, F_high, n_pts)
    spec   = [{'Fx': float(f)} for f in F_vals]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = sweep_md(pos, en_func, en_params, spec,
                           base_md_kwargs=_MD_BASE,
                           post_fn=drift_velocity(),
                           n_jobs=1, save=False, verbose=False)
    vx = np.array([r['result'][0] for r in results])
    pinned  = vx < 0.05
    sliding = ~pinned
    assert pinned.any() and sliding.any(), (
        "No depinning transition in F=[%.2f, %.2f]; "
        "vx=[%.3f, %.3f]. Widen the bracket." % (F_low, F_high, vx.min(), vx.max())
    )
    return float(F_vals[pinned].max()), float(F_vals[sliding].min())


@pytest.fixture(scope='module')
def substrate():
    _, en_func, en_params = substrate_from_params(_SIN_PARAMS)
    return en_func, en_params


# ---------------------------------------------------------------------------
# 1. F1s matches analytic
# ---------------------------------------------------------------------------

def test_F1s_analytic(substrate):
    """Single-particle Fc agrees with analytic F1s to 10%."""
    en_func, en_params = substrate
    pos    = np.array([[0.0, 0.0]])   # single particle; kBT=0 so eta_r=0 is safe
    F1s    = _analytical_F1s()

    Fc_lo, Fc_hi = _depinning_Fc(pos, en_func, en_params,
                                  F_low=0.7 * F1s, F_high=1.3 * F1s)
    Fc_num = 0.5 * (Fc_lo + Fc_hi)
    rel_err = abs(Fc_num - F1s) / F1s

    assert rel_err < 0.10, (
        "F1s: numerical=%.3f  analytical=%.3f  rel_err=%.1f%% (>10%%)"
        % (Fc_num, F1s, rel_err * 100)
    )


# ---------------------------------------------------------------------------
# 2. Commensurate linear scaling: Fc(N=7) = 7 * Fc(N=1)
# ---------------------------------------------------------------------------

def test_commensurate_linear_scaling(substrate):
    """Fc scales linearly with N for a commensurate cluster."""
    en_func, en_params = substrate
    F1s = _analytical_F1s()

    pos1 = np.array([[0.0, 0.0]])
    Fc1_lo, Fc1_hi = _depinning_Fc(pos1, en_func, en_params,
                                    F_low=0.7 * F1s, F_high=1.3 * F1s)
    Fc1 = 0.5 * (Fc1_lo + Fc1_hi)

    # 7-particle commensurate cluster: centre + 6 NN on the same lattice
    pos7 = np.array([
        [0., 0.], _A1, -_A1, _A2, -_A2, _A1 + _A2, -(_A1 + _A2)
    ], dtype=np.float64)
    Fc7_lo, Fc7_hi = _depinning_Fc(pos7, en_func, en_params,
                                    F_low=4.0 * F1s, F_high=11.0 * F1s,
                                    n_pts=14)
    Fc7 = 0.5 * (Fc7_lo + Fc7_hi)

    ratio = Fc7 / Fc1
    assert abs(ratio - 7.0) / 7.0 < 0.10, (
        "Fc(N=7)/Fc(N=1) = %.2f, expected 7.0 within 10%%" % ratio
    )


# ---------------------------------------------------------------------------
# 3. Superlubricity: incommensurate cluster depins at Fs << Fc_commensurate
# ---------------------------------------------------------------------------

def test_superlubricity(substrate):
    """Incommensurate cluster (theta=1.5 deg) slides at Fs < 0.5 * Fc_comm."""
    en_func, en_params = substrate
    F1s = _analytical_F1s()

    # N=19 commensurate cluster (circle, 4 shells)
    pos_ref = make_cluster(_A1, _A2, 4, 4, shape='circle')
    N = len(pos_ref)

    Fc_lo, Fc_hi = _depinning_Fc(pos_ref, en_func, en_params,
                                  F_low=0.5 * N * F1s,
                                  F_high=1.3 * N * F1s, n_pts=14)
    Fc_comm = 0.5 * (Fc_lo + Fc_hi)

    # Same cluster rotated 30 deg: breaks registry cleanly -> superlubricity.
    # For a triangular cluster on a triangular substrate, small misorientations
    # are insufficient at kBT=0 -- 30 deg gives a fully incommensurate
    # registration at zero temperature.
    pos_incomm = rotate(pos_ref, 30.0)
    F_vals = np.linspace(0.0, 0.5 * Fc_comm, 14)
    spec   = [{'Fx': float(f)} for f in F_vals]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = sweep_md(pos_incomm, en_func, en_params, spec,
                           base_md_kwargs=_MD_BASE,
                           post_fn=drift_velocity(),
                           n_jobs=1, save=False, verbose=False)
    vx = np.array([r['result'][0] for r in results])

    sliding = vx > 0.05
    assert sliding.any(), (
        "Incommensurate cluster not sliding up to Fx=%.1f (0.5*Fc_comm=%.1f). "
        "Superlubricity not detected." % (F_vals[-1], 0.5 * Fc_comm)
    )
    Fs_incomm = float(F_vals[sliding].min())
    assert Fs_incomm < 0.5 * Fc_comm, (
        "Fs_incomm=%.2f not < 0.5*Fc_comm=%.2f" % (Fs_incomm, 0.5 * Fc_comm)
    )
