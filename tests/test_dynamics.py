"""
Tests for dynamics.py (overdamped Langevin MD).

Substrate used throughout: sinusoidal triangular (n=3, epsilon=1, spacing=1),
same as test_maps.py.  The 7-particle commensurate cluster and the flat-
substrate helper are defined locally so this file is self-contained.

Flat substrate helper
---------------------
_flat_sub(abs_pos, pos_cm) returns (0.0, zeros(2), 0.0) and accepts no
extra en_params.  run_md is called with en_params=[] so the *en_params
unpacking passes nothing.

Cluster choices
---------------
SINGLE: [[0,0]] -- used for purely translational tests (kBT=0, Tau=0).
                   eta_r=0 is acceptable when kBT=0 and Tau=0.
TWO_PART: two particles on a rod -- used for noise-amplitude test where
          kBT>0; having eta_r>0 is required so the rotational update does
          not divide by zero.
_COMM_POS: 7-particle commensurate cluster for pinning test.
"""

import numpy as np
import pytest

from tool_create_substrate import substrate_from_params, get_ks
from tool_create_cluster import calc_cluster_langevin
from dynamics import run_md, make_params_array


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

A1 = np.array([1.0, 0.0])
A2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

SIN_PARAMS = {
    'well_shape': 'sin',
    'epsilon':    1.0,
    'sub_basis':  [[0., 0.]],
    'ks':         get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist(),
}

# 7-particle commensurate cluster (center + 6 nearest neighbours).
_COMM_POS = np.array([
    [0., 0.], A1, -A1, A2, -A2, A1 + A2, -(A1 + A2)
], dtype=np.float64)

# Single particle at origin -- eta_r=0; only valid when kBT=0 and Tau=0.
SINGLE = np.array([[0., 0.]])

# Two-particle rod: eta_r = eta * 2 * (0.5)^2 = 0.5*eta > 0.
# Used for tests that require kBT > 0.
TWO_PART = np.array([[0.5, 0.], [-0.5, 0.]])


def _flat_sub(abs_pos, pos_cm):
    """Flat (zero) substrate: no energy, force, or torque."""
    return 0.0, np.zeros(2, dtype=np.float64), 0.0


@pytest.fixture(scope='module')
def substrate():
    """Return (calc_en_f, en_params) for the triangular sin substrate."""
    _, calc_en_f, en_params = substrate_from_params(SIN_PARAMS)
    return calc_en_f, en_params


# ---------------------------------------------------------------------------
# 1. Noise amplitude
# ---------------------------------------------------------------------------

def test_noise_amplitude_correct():
    """var(delta_x) == 2*D_t*dt within 20% (TWO_PART cluster, kBT=1)."""
    eta   = 1.0
    kBT   = 1.0
    dt    = 1e-3
    n     = 50000
    eta_t, _ = calc_cluster_langevin(eta, TWO_PART)
    D_t = kBT / eta_t

    result = run_md(TWO_PART, _flat_sub, [], eta=eta, kBT=kBT,
                    dt=dt, n_steps=n, print_every=1, seed=42)

    dx      = np.diff(result['pos_cm'][:, 0])
    var_x   = float(np.var(dx))
    expected = 2.0 * D_t * dt

    assert abs(var_x - expected) / expected < 0.20, (
        "var(dx)=%.4e, expected=%.4e (ratio=%.3f)"
        % (var_x, expected, var_x / expected)
    )


# ---------------------------------------------------------------------------
# 2. Zero temperature, no force -- particle stays put
# ---------------------------------------------------------------------------

def test_zero_temperature_no_diffusion():
    """kBT=0 and no external force: particle must not move at all."""
    result = run_md(SINGLE, _flat_sub, [], eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=1000, print_every=1, seed=0)

    pos = result['pos_cm']
    assert np.allclose(pos, 0., atol=1e-14), (
        "max displacement = %.2e" % np.max(np.abs(pos))
    )


# ---------------------------------------------------------------------------
# 3. Constant external force, flat substrate, zero temperature
# ---------------------------------------------------------------------------

def test_external_force_free_particle():
    """kBT=0, flat substrate, Fx: pos_cm_x = Fx/eta_t * t up to 1e-8."""
    eta   = 1.0
    Fx    = 0.5
    dt    = 1e-3
    n     = 2000
    eta_t, _ = calc_cluster_langevin(eta, SINGLE)

    result = run_md(SINGLE, _flat_sub, [], eta=eta, Fx=Fx, kBT=0.,
                    dt=dt, n_steps=n, print_every=1, seed=0)

    t_arr  = result['t']
    x_arr  = result['pos_cm'][:, 0]
    x_pred = Fx / eta_t * t_arr

    assert np.allclose(x_arr, x_pred, atol=1e-8), (
        "max error = %.2e" % np.max(np.abs(x_arr - x_pred))
    )


# ---------------------------------------------------------------------------
# 4. Energy decreases monotonically to minimum (kBT=0)
# ---------------------------------------------------------------------------

def test_energy_decreases_to_minimum(substrate):
    """kBT=0, particle starts at (0.3,0): energy must decrease monotonically."""
    calc_en_f, en_params = substrate

    result = run_md(SINGLE, calc_en_f, en_params, eta=1.0, kBT=0.,
                    dt=1e-4, n_steps=5000, print_every=1,
                    pos_cm0=np.array([0.3, 0.0]), seed=0)

    energy = result['energy']
    diffs  = np.diff(energy)
    assert np.all(diffs <= 1e-12), (
        "energy increased at %d steps; max increase = %.2e"
        % (np.sum(diffs > 1e-12), float(diffs.max()))
    )


# ---------------------------------------------------------------------------
# 5. stop_fn early termination
# ---------------------------------------------------------------------------

def test_stop_fn():
    """stop_fn(step, state_dict) returns True at step 50; length <= 50//pe + 1."""
    print_every  = 10
    trigger_step = 50

    def my_stop(step, state_dict):
        return step >= trigger_step

    result = run_md(SINGLE, _flat_sub, [], eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=10000,
                    print_every=print_every,
                    stop_fn=my_stop, seed=0)

    # Records at steps 10, 20, 30, 40, 50 (then stop) = 5; +1 is loose bound.
    max_records = trigger_step // print_every + 1
    assert len(result['t']) <= max_records, (
        "expected <= %d records, got %d" % (max_records, len(result['t']))
    )


# ---------------------------------------------------------------------------
# 6. Commensurate cluster stays pinned under small external force
# ---------------------------------------------------------------------------

def test_commensurate_cluster_pinned(substrate):
    """kBT=0, Fx=0.01*epsilon: 7-particle commensurate cluster must not slide."""
    calc_en_f, en_params = substrate
    epsilon = SIN_PARAMS['epsilon']
    Fx      = 0.01 * epsilon   # well below the translational barrier

    result = run_md(_COMM_POS, calc_en_f, en_params, eta=1.0, Fx=Fx, kBT=0.,
                    dt=1e-4, n_steps=5000, print_every=100, seed=0)

    final_x = abs(float(result['pos_cm'][-1, 0]))
    assert final_x < 0.5, (
        "cluster slid to x=%.4f under Fx=%.3f" % (final_x, Fx)
    )


# ---------------------------------------------------------------------------
# 7. Output dict keys and array shapes
# ---------------------------------------------------------------------------

def test_output_dict_keys():
    """All expected keys present and shapes consistent with n_rec."""
    n_steps     = 100
    print_every = 10

    result = run_md(SINGLE, _flat_sub, [], eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=n_steps,
                    print_every=print_every, seed=0)

    expected_keys = {'t', 'pos_cm', 'theta', 'energy', 'force',
                     'torque', 'vel_cm', 'omega'}
    assert expected_keys == set(result.keys())

    n_rec = len(result['t'])
    assert result['pos_cm'].shape == (n_rec, 2)
    assert result['theta'].shape  == (n_rec,)
    assert result['energy'].shape == (n_rec,)
    assert result['force'].shape  == (n_rec, 2)
    assert result['torque'].shape == (n_rec,)
    assert result['vel_cm'].shape == (n_rec, 2)
    assert result['omega'].shape  == (n_rec,)


# ---------------------------------------------------------------------------
# 8. output_fn callback: run_md returns None; callback receives all records
# ---------------------------------------------------------------------------

def test_output_fn_no_return():
    """output_fn receives every snapshot; run_md returns None."""
    n_steps     = 100
    print_every = 10
    collected   = []

    def my_output(step, t, state_dict):
        collected.append({'step': step, 't': t, **state_dict})

    result = run_md(SINGLE, _flat_sub, [], eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=n_steps,
                    print_every=print_every,
                    output_fn=my_output, seed=0)

    assert result is None
    assert len(collected) == n_steps // print_every


# ---------------------------------------------------------------------------
# 9. Near-zero temperature: particle relaxes toward minimum
# ---------------------------------------------------------------------------

def test_gradient_descent_at_low_kBT(substrate):
    """kBT=1e-8 (near-zero): cluster starting at (0.3,0) relaxes toward minimum."""
    calc_en_f, en_params = substrate
    pos_cm0 = np.array([0.3, 0.0])

    # TWO_PART has eta_r > 0, required when kBT > 0.
    r = run_md(TWO_PART, calc_en_f, en_params, eta=1.0, kBT=1e-8,
               dt=1e-4, n_steps=5000, print_every=1,
               pos_cm0=pos_cm0.copy(), seed=0)

    assert r['energy'][-1] < r['energy'][0], (
        "Energy did not decrease: E0=%.6f, E_final=%.6f"
        % (r['energy'][0], r['energy'][-1])
    )
