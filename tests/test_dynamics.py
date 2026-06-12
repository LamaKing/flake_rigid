"""
Tests for dynamics.py (overdamped Langevin MD).

Substrate used throughout: sinusoidal triangular (n=3, epsilon=1, spacing=1),
same as test_maps.py.  The 7-particle commensurate cluster and the flat-
substrate helper are defined locally so this file is self-contained.

Flat substrate
--------------
_flat_sub(abs_pos, pos_cm) is a plain Python callable -- no _jit_core.
It is used only in tests that pass stop_fn or output_fn (Python loop path).

For tests without callbacks (JIT path), flat_sub_jit is used: a sin
substrate with epsilon=1e-300 so forces are zero to machine precision but
the en_func has _jit_core and _jit_params attached.

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

from flake.substrate import substrate_from_params, get_ks, _calc_en_flat_core
from flake.cluster import calc_cluster_langevin
from flake.dynamics import run_md


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

_FLAT_PARAMS = {'well_shape': 'flat'}

# Gaussian substrate on a square lattice spacing=1.
# sigma=0.2, a=0.4, b=0.5 keep the well narrow relative to the unit cell,
# so a commensurate cluster sitting at well centres has energy ~ -N*epsilon
# and a clear barrier separating minima.
GAUSS_PARAMS = {
    'well_shape': 'gaussian',
    'epsilon':    1.0,
    'sigma':      0.2,
    'a':          0.4,
    'b':          0.5,
    'b1':         [1.0, 0.0],
    'b2':         [0.0, 1.0],
    'sub_basis':  [[0., 0.]],
}

# 5-particle commensurate cluster on the square lattice (centre + 4 NN).
_COMM_SQ = np.array([
    [0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]
], dtype=np.float64)

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
    """Flat (zero) substrate: plain Python callable, no _jit_core.

    Only valid for the Python loop path (stop_fn or output_fn present).
    """
    return 0.0, np.zeros(2, dtype=np.float64), 0.0


@pytest.fixture(scope='module')
def substrate():
    """Return (calc_en_f, en_params) for the triangular sin substrate."""
    _, calc_en_f, en_params = substrate_from_params(SIN_PARAMS)
    return calc_en_f, en_params


@pytest.fixture(scope='module')
def gauss_substrate():
    """Return (calc_en_f, en_params) for the square-lattice Gaussian substrate."""
    _, calc_en_f, en_params = substrate_from_params(GAUSS_PARAMS)
    return calc_en_f, en_params


@pytest.fixture(scope='module')
def flat_sub_jit():
    """Return (calc_en_f, en_params) for the flat (zero) substrate.

    Energy and force are identically zero; _jit_core/_jit_params are present.
    Use in JIT-path tests that need a substrate-free integrator.
    """
    _, calc_en_f, en_params = substrate_from_params(_FLAT_PARAMS)
    return calc_en_f, en_params


# ---------------------------------------------------------------------------
# 1. Noise amplitude
# ---------------------------------------------------------------------------

def test_noise_amplitude_correct(flat_sub_jit):
    """var(delta_x) == 2*D_t*dt within 20% (TWO_PART cluster, kBT=1)."""
    calc_en_f, en_params = flat_sub_jit
    eta   = 1.0
    kBT   = 1.0
    dt    = 1e-3
    n     = 50000
    eta_t, _ = calc_cluster_langevin(eta, TWO_PART)
    D_t = kBT / eta_t

    result = run_md(TWO_PART, calc_en_f, en_params, eta=eta, kBT=kBT,
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

def test_zero_temperature_no_diffusion(flat_sub_jit):
    """kBT=0 and no external force: particle must not move at all."""
    calc_en_f, en_params = flat_sub_jit
    result = run_md(SINGLE, calc_en_f, en_params, eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=1000, print_every=1, seed=0)

    pos = result['pos_cm']
    assert np.allclose(pos, 0., atol=1e-14), (
        "max displacement = %.2e" % np.max(np.abs(pos))
    )


# ---------------------------------------------------------------------------
# 3. Constant external force, flat substrate, zero temperature
# ---------------------------------------------------------------------------

def test_external_force_free_particle(flat_sub_jit):
    """kBT=0, flat substrate, Fx: pos_cm_x = Fx/eta_t * t up to 1e-8."""
    calc_en_f, en_params = flat_sub_jit
    eta   = 1.0
    Fx    = 0.5
    dt    = 1e-3
    n     = 2000
    eta_t, _ = calc_cluster_langevin(eta, SINGLE)

    result = run_md(SINGLE, calc_en_f, en_params, eta=eta, Fx=Fx, kBT=0.,
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

def test_output_dict_keys(flat_sub_jit):
    """All expected keys present and shapes consistent with n_rec."""
    calc_en_f, en_params = flat_sub_jit
    n_steps     = 100
    print_every = 10

    result = run_md(SINGLE, calc_en_f, en_params, eta=1.0, kBT=0.,
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


# ---------------------------------------------------------------------------
# 10. JIT path: no force, kBT=0 -- particle stays at origin
# ---------------------------------------------------------------------------

def test_jit_no_force_no_motion(flat_sub_jit):
    """JIT path: kBT=0, no external force -- CM stays at origin."""
    calc_en_f, en_params = flat_sub_jit
    result = run_md(TWO_PART, calc_en_f, en_params, eta=1.0, kBT=0.,
                    dt=1e-3, n_steps=200, print_every=10, seed=42)
    assert np.allclose(result['pos_cm'], 0., atol=1e-14), (
        "max displacement = %.2e" % np.max(np.abs(result['pos_cm']))
    )


# ---------------------------------------------------------------------------
# 11. JIT path: noise amplitude matches FDT
# ---------------------------------------------------------------------------

def test_jit_noise_amplitude(flat_sub_jit):
    """JIT path: var(delta_x) == 2*D_t*dt within 20%."""
    calc_en_f, en_params = flat_sub_jit
    eta = 1.0
    kBT = 1.0
    dt  = 1e-3
    n   = 50000
    eta_t, _ = calc_cluster_langevin(eta, TWO_PART)
    D_t = kBT / eta_t

    result = run_md(TWO_PART, calc_en_f, en_params, eta=eta, kBT=kBT,
                    dt=dt, n_steps=n, print_every=1, seed=42)
    dx  = np.diff(result['pos_cm'][:, 0])
    var = float(np.var(dx))
    assert abs(var - 2. * D_t * dt) / (2. * D_t * dt) < 0.20, (
        "var(dx)=%.4e, expected=%.4e" % (var, 2. * D_t * dt)
    )


# ---------------------------------------------------------------------------
# 12. Missing _jit_core raises NotImplementedError on JIT path
# ---------------------------------------------------------------------------

def test_no_jit_core_raises():
    """run_md without callbacks raises NotImplementedError for a plain callable."""
    with pytest.raises(NotImplementedError, match="_jit_core"):
        run_md(SINGLE, _flat_sub, [], eta=1.0, kBT=0.,
               dt=1e-3, n_steps=10, print_every=5, seed=0)


# ---------------------------------------------------------------------------
# 13. Flat substrate: analytic checks (drift velocity, diffusion)
# ---------------------------------------------------------------------------

def test_flat_drift_velocity(flat_sub_jit):
    """F/eta_t * t analytic prediction matches JIT trajectory to 1e-8."""
    calc_en_f, en_params = flat_sub_jit
    eta = 1.0
    Fx  = 0.5
    dt  = 1e-3
    n   = 2000
    eta_t, _ = calc_cluster_langevin(eta, TWO_PART)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = run_md(TWO_PART, calc_en_f, en_params, eta=eta, Fx=Fx, kBT=0.,
                   dt=dt, n_steps=n, print_every=1, seed=0)

    x_pred = Fx / eta_t * r['t']
    assert np.allclose(r['pos_cm'][:, 0], x_pred, atol=1e-8)


def test_flat_diffusion(flat_sub_jit):
    """var(delta_x) == 2*D_t*dt to 5% on flat substrate (N=50000 steps)."""
    calc_en_f, en_params = flat_sub_jit
    eta = 1.0
    kBT = 1.0
    dt  = 1e-3
    n   = 50000
    eta_t, _ = calc_cluster_langevin(eta, TWO_PART)
    D_t = kBT / eta_t

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = run_md(TWO_PART, calc_en_f, en_params, eta=eta, kBT=kBT,
                   dt=dt, n_steps=n, print_every=1, seed=7)

    dx  = np.diff(r['pos_cm'][:, 0])
    var = float(np.var(dx))
    assert abs(var - 2. * D_t * dt) / (2. * D_t * dt) < 0.05, (
        "var(dx)=%.4e expected=%.4e" % (var, 2. * D_t * dt)
    )


# ---------------------------------------------------------------------------
# 14. JIT speedup benchmark  (pytest -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_jit_speedup_benchmark():
    """Compare Python-loop vs JIT-loop integration paths.

    Two paths on an N=85 commensurate sin substrate:

      Python-loop: stop_fn=always-False forces the Python EM loop.
        Every step crosses the Python/JIT boundary once to call the
        substrate @njit core, then returns to Python for RNG, rotate,
        state dict, and the loop itself.  This is the old code pattern.

      JIT-loop: no callbacks.  The entire EM loop -- rotate, substrate
        call, RNG, state update -- runs inside a single @njit function.
        Zero Python/JIT boundary crossings per step.

    The speedup from the JIT loop comes entirely from eliminating the
    Python overhead per step (~15 us fixed cost: rotate, dict, RNG on the
    Python side).  At small N this overhead dominates, giving ~4x speedup.
    At large N the O(N) substrate cost dominates and the relative gain
    shrinks (~2x at N=397), but the absolute saving per step is the same.

    Warmup runs are done first so compilation cost is excluded from timing.
    n_steps=500000 at dt=5e-4 (~250 physical time units).

    The test only asserts speedup > 1 -- the exact ratio depends on N and
    machine load and should be read from the printed table, not a threshold.

    Run with:  pytest tests/test_dynamics.py -m slow -v -s
    """
    import warnings
    from time import perf_counter

    A1 = np.array([1.0, 0.0])
    A2 = np.array([-0.5, np.sqrt(3.0) / 2.0])

    from flake.cluster import make_cluster
    pos = make_cluster(A1, A2, 9, 10, shape='circle')   # N=85
    N   = len(pos)

    sin_params = {
        'well_shape': 'sin', 'epsilon': 1.0,
        'sub_basis': [[0., 0.]],
        'ks': get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist(),
    }
    _, sin_en_f, sin_en_p = substrate_from_params(sin_params)

    n_steps = 500_000
    md_kw = dict(eta=1.0, kBT=1e-5, Fx=0.3, dt=5e-4,
                 n_steps=n_steps, print_every=n_steps, seed=42)

    def _never_stop(step, state):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # warmup: compile JIT paths before timing
        wkw = dict(md_kw, n_steps=200, print_every=200)
        run_md(pos, sin_en_f, sin_en_p, **wkw)
        run_md(pos, sin_en_f, sin_en_p, stop_fn=_never_stop, **wkw)

        # Python-loop: stop_fn forces _run_python_loop
        t0 = perf_counter()
        run_md(pos, sin_en_f, sin_en_p, stop_fn=_never_stop, **md_kw)
        t_python = perf_counter() - t0

        # JIT-loop: no callbacks
        t0 = perf_counter()
        run_md(pos, sin_en_f, sin_en_p, **md_kw)
        t_jit = perf_counter() - t0

    speedup = t_python / t_jit
    print("\n--- JIT speedup benchmark  N=%d  n_steps=%d ---" % (N, n_steps))
    print("Python-loop (stop_fn): %.2f us/step  (%.2f s)" % (t_python * 1e6 / n_steps, t_python))
    print("JIT-loop    (no cb)  : %.2f us/step  (%.2f s)" % (t_jit    * 1e6 / n_steps, t_jit))
    print("speedup: %.2fx" % speedup)
    print("Note: speedup shrinks with N because O(N) substrate cost")
    print("dominates at large N; the ~15 us/step Python overhead is fixed.")

    assert speedup > 1.0, (
        "JIT-loop (%.2f s) not faster than Python-loop (%.2f s)" % (t_jit, t_python)
    )


# ---------------------------------------------------------------------------
# Gaussian substrate: energy descent, pinning, JIT speedup
# ---------------------------------------------------------------------------

def test_gauss_energy_decreases(gauss_substrate):
    """kBT=0, single particle starts off-minimum: energy must decrease monotonically."""
    calc_en_f, en_params = gauss_substrate

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run_md(SINGLE, calc_en_f, en_params, eta=1.0, kBT=0.,
                        dt=1e-4, n_steps=5000, print_every=1,
                        pos_cm0=np.array([0.3, 0.0]), seed=0)

    diffs = np.diff(result['energy'])
    assert np.all(diffs <= 1e-12), (
        "energy increased at %d steps; max increase = %.2e"
        % (np.sum(diffs > 1e-12), float(diffs.max()))
    )


def test_gauss_commensurate_pinned(gauss_substrate):
    """kBT=0, Fx=0.01*epsilon: 5-particle commensurate cluster stays pinned."""
    calc_en_f, en_params = gauss_substrate
    Fx = 0.01 * GAUSS_PARAMS['epsilon']

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run_md(_COMM_SQ, calc_en_f, en_params, eta=1.0, Fx=Fx, kBT=0.,
                        dt=1e-4, n_steps=5000, print_every=100, seed=0)

    final_x = abs(float(result['pos_cm'][-1, 0]))
    assert final_x < 0.5, (
        "cluster slid to x=%.4f under Fx=%.3f" % (final_x, Fx)
    )


@pytest.mark.slow
def test_gauss_jit_speedup_benchmark():
    """JIT-loop faster than Python-loop on Gaussian substrate (N=85).

    Mirrors test_jit_speedup_benchmark but uses the Gaussian well to confirm
    that _calc_en_gaussian_core compiles and dispatches correctly via _jit_params.
    """
    import warnings
    from time import perf_counter
    from flake.cluster import make_cluster

    pos = make_cluster(A1, A2, 9, 10, shape='circle')   # N=85
    N   = len(pos)

    _, gauss_en_f, gauss_en_p = substrate_from_params(GAUSS_PARAMS)

    n_steps = 500_000
    md_kw = dict(eta=1.0, kBT=1e-5, Fx=0.3, dt=5e-4,
                 n_steps=n_steps, print_every=n_steps, seed=42)

    def _never_stop(step, state):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        wkw = dict(md_kw, n_steps=200, print_every=200)
        run_md(pos, gauss_en_f, gauss_en_p, **wkw)
        run_md(pos, gauss_en_f, gauss_en_p, stop_fn=_never_stop, **wkw)

        t0 = perf_counter()
        run_md(pos, gauss_en_f, gauss_en_p, stop_fn=_never_stop, **md_kw)
        t_python = perf_counter() - t0

        t0 = perf_counter()
        run_md(pos, gauss_en_f, gauss_en_p, **md_kw)
        t_jit = perf_counter() - t0

    speedup = t_python / t_jit
    print("\n--- Gaussian JIT speedup  N=%d  n_steps=%d ---" % (N, n_steps))
    print("Python-loop (stop_fn): %.2f us/step  (%.2f s)" % (t_python * 1e6 / n_steps, t_python))
    print("JIT-loop    (no cb)  : %.2f us/step  (%.2f s)" % (t_jit    * 1e6 / n_steps, t_jit))
    print("speedup: %.2fx" % speedup)

    assert speedup > 1.0, (
        "JIT-loop (%.2f s) not faster than Python-loop (%.2f s)" % (t_jit, t_python)
    )
