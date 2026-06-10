"""
Parameter sweeps over MD simulations for rigid-cluster systems.

This module is the time-dependent analogue of maps.py: instead of scanning
a static energy landscape, it runs one full MD trajectory per parameter
point and collects observables.

Public API
----------
    sweep_md       -- run one MD simulation per point in a sweep spec.
    grid_sweep     -- Cartesian-product sweep spec from axis ranges.
    line_sweep     -- explicit point-by-point sweep spec from a column table.
    force_sweep    -- sweep force magnitude at a fixed angle.
    concat_sweeps  -- merge multiple specs, removing duplicates.
    last_state     -- post_fn factory: extract last snapshot.
    mean_velocity  -- post_fn factory: mean |v_cm| over a tail window.
    drift_velocity -- post_fn factory: net CM displacement / total time.
    load_sweep     -- reload a saved sweep from disk.
    filter_sweep   -- remove None-result entries from load_sweep output.

I/O convention
--------------
Each run writes two files:
    outdir/run_NNNN/traj.h5    -- trajectory arrays (HDF5 via h5py)
    outdir/run_NNNN/params.yaml -- run parameters (YAML via PyYAML)

Neither h5py nor yaml is imported at module level; both are guarded
inside the functions that use them.
"""

import os
import itertools
import warnings
import logging
import numpy as np

from drift.dynamics import run_md

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

# Valid override keys for run_md (everything after en_params in its signature).
_VALID_MD_KWARGS = frozenset({
    'eta', 'Fx', 'Fy', 'Tau', 'kBT', 'dt', 'n_steps',
    'theta0', 'pos_cm0', 'print_every', 'stop_fn', 'output_fn', 'seed',
})


# ============================================================
# Internal helpers
# ============================================================

def _dict_key(d):
    """Convert a param dict to a hashable key for duplicate detection.

    Numpy arrays are reduced to their raw bytes so equality is content-based.
    Callables fall back to object identity (they rarely duplicate in practice).
    """
    items = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, np.ndarray):
            items.append((k, v.tobytes()))
        elif isinstance(v, (list, tuple)):
            items.append((k, repr(v)))
        elif callable(v):
            items.append((k, id(v)))
        else:
            items.append((k, v))
    return frozenset(items)


def _dedup(spec):
    """Return (deduped_list, n_removed) with order preserved."""
    seen = set()
    out = []
    n_removed = 0
    for point in spec:
        key = _dict_key(point)
        if key not in seen:
            seen.add(key)
            out.append(point)
        else:
            n_removed += 1
    return out, n_removed


def _yaml_safe(params):
    """Strip non-serialisable entries and convert numpy arrays to lists."""
    out = {}
    for k, v in params.items():
        if callable(v):
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            # numpy scalar (e.g. np.float64) -> plain Python
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ============================================================
# I/O helpers
# ============================================================

def _save_run(run_dir, traj_dict, run_params, save_traj=True):
    """Save one run to run_dir.

    Always writes params.yaml.  Writes traj.h5 only when save_traj=True
    and traj_dict is not None.

    Args:
        run_dir:    str  -- path to the run directory.
        traj_dict:  dict or None -- trajectory arrays from run_md.
        run_params: dict -- full kwarg dict used for this run.
        save_traj:  bool -- if True (default), write traj.h5.
    """
    try:
        from drift.slides_io import save_trajectory, save_params
    except ImportError:
        raise ImportError(
            "slides_io is required for _save_run. "
            "Ensure slides_io.py is on the Python path."
        )

    os.makedirs(run_dir, exist_ok=True)

    if save_traj and traj_dict is not None:
        save_trajectory(traj_dict, os.path.join(run_dir, 'traj.h5'))

    save_params(_yaml_safe(run_params), os.path.join(run_dir, 'params.yaml'))


def _load_run(run_dir):
    """Load one run from run_dir.

    Args:
        run_dir: str -- path to a run_NNNN directory.

    Returns:
        (traj_dict, run_params) -- trajectory arrays and parameter dict.
    """
    try:
        from drift.slides_io import load_trajectory, load_params
    except ImportError:
        raise ImportError(
            "slides_io is required for _load_run."
        )

    traj_path   = os.path.join(run_dir, 'traj.h5')
    params_path = os.path.join(run_dir, 'params.yaml')

    traj_dict = None
    if os.path.isfile(traj_path):
        traj_dict, _ = load_trajectory(traj_path)

    run_params = {}
    if os.path.isfile(params_path):
        run_params = load_params(params_path)

    return traj_dict, run_params


def load_sweep(outdir):
    """Load all runs from a sweep output directory.

    Scans for run_NNNN/ subdirectories in numeric order.  Every directory
    that matches the pattern produces one entry in the returned list -- runs
    are never silently dropped.  Missing or incomplete runs produce an entry
    with result=None and a WARNING; the caller is responsible for filtering
    (see filter_sweep).

    Args:
        outdir: str -- directory written by sweep_md with save=True.

    Returns:
        list of dict, one per run_NNNN directory found, each with keys:
            'params':  dict of run parameters, or None if params.yaml absent.
            'result':  full traj dict, or None if traj.h5 absent.
            'run_dir': absolute path to the run_NNNN directory.

    Example:
        raw     = load_sweep('output/my_sweep')
        results = filter_sweep(raw)
        for r in results:
            print(r['params']['Fx'], r['result']['energy'][-1])
    """
    import re
    try:
        from drift.slides_io import load_trajectory, load_params
    except ImportError:
        raise ImportError("slides_io is required for load_sweep.")

    pattern = re.compile(r'^run_(\d{4})$')

    entries = sorted(
        d for d in os.listdir(outdir)
        if pattern.match(d) and os.path.isdir(os.path.join(outdir, d))
    )

    results = []
    n_ok = n_missing_traj = n_bad = 0

    for entry in entries:
        run_dir     = os.path.join(outdir, entry)
        params_path = os.path.join(run_dir, 'params.yaml')
        traj_path   = os.path.join(run_dir, 'traj.h5')

        if not os.path.isfile(params_path):
            warnings.warn(
                "%s: params.yaml not found; params and result will be None."
                % entry,
                UserWarning, stacklevel=2
            )
            results.append({'params': None, 'result': None, 'run_dir': run_dir})
            n_bad += 1
            continue

        run_params = load_params(params_path)

        if not os.path.isfile(traj_path):
            warnings.warn(
                "%s: traj.h5 not found; result will be None." % entry,
                UserWarning, stacklevel=2
            )
            results.append({'params': run_params, 'result': None,
                            'run_dir': run_dir})
            n_missing_traj += 1
            continue

        traj_dict, _ = load_trajectory(traj_path)
        results.append({'params': run_params, 'result': traj_dict,
                        'run_dir': run_dir})
        n_ok += 1

    n_total = len(entries)
    summary = (
        "load_sweep: loaded %d complete runs, %d missing traj, "
        "%d missing params, out of %d directories found."
        % (n_ok, n_missing_traj, n_bad, n_total)
    )
    _log.info(summary)
    print(summary, flush=True)

    return results


def filter_sweep(loaded, warn_fraction=0.1):
    """Remove None-result entries from load_sweep output.

    Emits a WARNING if the fraction of None entries exceeds warn_fraction
    (default 10%).

    Args:
        loaded:        list from load_sweep.
        warn_fraction: float -- threshold for the warning.

    Returns:
        list with None-result entries removed.
    """
    n_total  = len(loaded)
    filtered = [r for r in loaded if r['result'] is not None]
    n_none   = n_total - len(filtered)
    if n_total > 0 and (n_none / n_total) > warn_fraction:
        warnings.warn(
            "filter_sweep: %d of %d runs (%.0f%%) have result=None. "
            "Check sweep output for incomplete or crashed runs."
            % (n_none, n_total, 100. * n_none / n_total),
            UserWarning, stacklevel=2
        )
    return filtered


# ============================================================
# Grid construction helpers
# ============================================================

def grid_sweep(axes):
    """Cartesian product of parameter ranges.

    Args:
        axes: dict mapping parameter name -> list or array of values.
              Example: {'Fx': [0.0, 0.1, 0.2], 'Tau': [0.0, 1.0]}
              gives 6 points covering all (Fx, Tau) combinations.

    Returns:
        list of dict, one per grid point.

    Example:
        spec = grid_sweep({'Fx': np.linspace(0, 0.5, 10),
                           'Tau': [0.0, 1.0]})
    """
    keys   = list(axes.keys())
    ranges = [list(axes[k]) for k in keys]
    return [dict(zip(keys, vals)) for vals in itertools.product(*ranges)]


def line_sweep(table):
    """Explicit list of points from a column table.

    Each row is one run; each key is one column.  Equivalent to the old
    'setf' file approach: load a text file, pass the columns as arrays.

    Args:
        table: dict mapping parameter name -> list or array of values.
               All lists must have the same length.
               Example: {'Tau': [0, 1, 2], 'Fx': [0, 0, 0]}
               gives 3 points.

    Returns:
        list of dict, one per row.

    Raises:
        ValueError: if lists have different lengths.

    Example:
        data = np.loadtxt('setf.dat')  # columns: Tau, Fx, Fy
        spec = line_sweep({'Tau': data[:,0], 'Fx': data[:,1], 'Fy': data[:,2]})
    """
    if not table:
        return []
    lengths = [len(v) for v in table.values()]
    if len(set(lengths)) != 1:
        raise ValueError(
            "All columns in table must have the same length; "
            "got lengths: %s" % dict(zip(table.keys(), lengths))
        )
    keys = list(table.keys())
    cols = [list(table[k]) for k in keys]
    return [dict(zip(keys, row)) for row in zip(*cols)]


def force_sweep(F_vals, phi_deg):
    """Sweep the external force magnitude at a fixed angle.

    Decomposes F into Fx = F * cos(phi), Fy = F * sin(phi).

    Args:
        F_vals:  array-like -- force magnitudes to sweep.
        phi_deg: float      -- force direction in degrees from x-axis.

    Returns:
        list of dict with keys 'Fx' and 'Fy'.

    Example:
        spec = force_sweep(np.linspace(0, 0.5, 20), phi_deg=30.0)
    """
    phi = np.deg2rad(float(phi_deg))
    cos_phi = float(np.cos(phi))
    sin_phi = float(np.sin(phi))
    return [{'Fx': float(F) * cos_phi, 'Fy': float(F) * sin_phi}
            for F in F_vals]


def concat_sweeps(*specs):
    """Concatenate multiple sweep specs, removing duplicates.

    Duplicates are detected by dict content (all key-value pairs equal).
    Order is preserved; later occurrences of a duplicate are dropped.

    Args:
        *specs: sweep spec lists (output of grid_sweep, line_sweep, etc.)

    Returns:
        list of dict.

    Example:
        spec = concat_sweeps(
            force_sweep(np.linspace(0, 0.3, 10), phi_deg=0.0),
            force_sweep(np.linspace(0, 0.3, 10), phi_deg=30.0),
        )
    """
    combined = [point for spec in specs for point in spec]
    deduped, _ = _dedup(combined)
    return deduped


# ============================================================
# post_fn helpers
# ============================================================

def last_state(keys=None):
    """Return a post_fn that extracts the last snapshot from the trajectory.

    Args:
        keys: list of str or None -- which traj_dict keys to extract.
              If None, extracts: 'pos_cm', 'theta', 'energy', 'vel_cm', 'omega'.

    Returns:
        callable post_fn(traj_dict, run_params) -> dict of last values.

    Example:
        results = sweep_md(..., post_fn=last_state(['pos_cm', 'theta']))
    """
    if keys is None:
        keys = ['pos_cm', 'theta', 'energy', 'vel_cm', 'omega']

    def _post_fn(traj_dict, run_params):
        return {k: traj_dict[k][-1] for k in keys}

    return _post_fn


def mean_velocity(fraction=0.2):
    """Return a post_fn that computes mean CM speed over the last fraction
    of the trajectory.

    Useful for pinned-vs-sliding detection: a pinned cluster has mean
    speed ~ 0; a sliding cluster has mean speed > 0.

    Args:
        fraction: float in (0, 1] -- use last (fraction * n_snapshots) steps.

    Returns:
        callable post_fn(traj_dict, run_params) -> float
        (mean of |vel_cm| over the window).

    Example:
        results = sweep_md(..., post_fn=mean_velocity(fraction=0.5))
    """
    def _post_fn(traj_dict, run_params):
        vel = traj_dict['vel_cm']
        n   = len(vel)
        # At least one snapshot must be in the window.
        start  = max(0, n - int(np.ceil(fraction * n)))
        speeds = np.linalg.norm(vel[start:], axis=1)
        return float(np.mean(speeds))

    return _post_fn


def drift_velocity():
    """Return a post_fn that computes the 2D drift velocity of the CM.

    Defined as (pos_cm_final - pos_cm_initial) / (t_final - t_initial).
    This is the physically meaningful quantity for translational depinning:
    zero vector when the cluster is pinned, nonzero above threshold.

    Returns:
        callable post_fn(traj_dict, run_params) -> (2,) float64 ndarray.

    Example:
        results = sweep_md(..., post_fn=drift_velocity())
        vdrift = np.array([r['result'] for r in results])  # shape (n_runs, 2)
    """
    def _post_fn(traj_dict, run_params):
        pos_cm = traj_dict['pos_cm']
        t      = traj_dict['t']
        dt_tot = float(t[-1] - t[0])
        if dt_tot == 0.0:
            return np.zeros(2, dtype=np.float64)
        return (pos_cm[-1] - pos_cm[0]) / dt_tot

    return _post_fn


# ============================================================
# Main sweep driver
# ============================================================

def sweep_md(pos, calc_en_f, en_params, sweep_spec,
             base_md_kwargs=None, post_fn=None,
             n_jobs=1, backend='loky', outdir='.', save=True, save_traj=True,
             verbose=True):
    """Run one MD simulation per point in sweep_spec.

    Each point in sweep_spec is a dict of keyword arguments that override
    base_md_kwargs for that run.  Keys absent from the point dict keep the
    base_md_kwargs value.  Keys not in run_md's signature raise ValueError
    at construction time (not at run time).

    All runs are independent (same starting state unless pos_cm0 is in the
    sweep spec).  For adiabatic continuation (each run starts where the
    previous ended), chain run_md calls manually.

    Args:
        pos:            (N, 2) ndarray -- cluster positions (reference frame).
        calc_en_f:      callable       -- substrate energy function.
        en_params:      list           -- extra arguments for calc_en_f.
        sweep_spec:     list of dict   -- one dict per run point.  Each dict
                                         overrides base_md_kwargs for that run.
                                         Duplicate points (same dict content)
                                         are silently removed after a warning.
        base_md_kwargs: dict or None   -- default keyword arguments for run_md.
                                         If None, run_md defaults are used.
                                         Do NOT include pos, calc_en_f,
                                         en_params here.
        post_fn:        callable or None
                        -- post_fn(traj_dict, run_params) -> dict or scalar.
                           Called after each run.  If None, the full trajectory
                           is stored in 'result'.
        n_jobs:         int  -- joblib parallel workers.  Do NOT set n_jobs > 1
                               if run_md itself already uses parallelism.
                               Default 1 (serial).
        backend:        str  -- joblib backend.  Default 'loky' (spawn-based
                               separate processes, no GIL).  Each worker
                               recompiles @njit functions (~1-2 s overhead),
                               so speedup requires runs of at least ~30 s each
                               to amortise that cost.  'threading' does NOT
                               help here: the EM loop has enough Python-level
                               work per step (numpy ops, RNG) that threads
                               serialize on the GIL and may be slower than
                               serial.
        outdir:         str  -- directory for per-run output.  Created if absent.
        save:           bool -- if True, write per-run output to outdir.
                               Each run writes:
                                 outdir/run_NNNN/traj.h5  (if save_traj=True)
                                 outdir/run_NNNN/params.yaml
        save_traj:      bool -- if True (default), write the full trajectory
                               to traj.h5 for each run.  Set False when
                               post_fn already extracts all needed information,
                               to avoid storing gigabytes of unused data.
                               params.yaml is always written.
        verbose:        bool -- if True (default), report progress.  Uses tqdm
                               if installed (works in TTY and batch logs alike);
                               otherwise prints one line per run to stdout.
                               Set False to suppress all progress output.

    Returns:
        list of dict, one per run point (in the same order as sweep_spec
        after deduplication).  Each dict contains:
            'params':  the full run_md kwargs used for this run.
            'result':  post_fn output if post_fn given, else the full traj dict.
            'run_dir': path to the run directory (if save=True, else None).

    Raises:
        ValueError: if any key in sweep_spec or base_md_kwargs is not a valid
                    run_md keyword argument.
        ImportError: if n_jobs != 1 and joblib is not installed.

    Example:
        spec = grid_sweep({'Fx': np.linspace(0, 0.5, 10), 'kBT': [1e-8]})
        results = sweep_md(pos, calc_en_f, en_params, spec,
                           base_md_kwargs={'eta': 1.0, 'n_steps': 50000},
                           post_fn=drift_velocity(),
                           n_jobs=4, backend='threading',
                           outdir='sweep_out')
    """
    pos = np.asarray(pos, dtype=np.float64)

    if base_md_kwargs is None:
        base_md_kwargs = {}

    # Validate all keys before launching any run.
    all_keys = set(base_md_kwargs.keys())
    for point in sweep_spec:
        all_keys.update(point.keys())
    invalid = all_keys - _VALID_MD_KWARGS
    if invalid:
        raise ValueError(
            "Invalid run_md keyword arguments: %s. "
            "Valid keys are: %s."
            % (sorted(invalid), sorted(_VALID_MD_KWARGS))
        )

    # Deduplicate.
    deduped, n_removed = _dedup(sweep_spec)
    if n_removed > 0:
        warnings.warn(
            "Removed %d duplicate point(s) from sweep_spec." % n_removed,
            UserWarning, stacklevel=2
        )

    if save:
        os.makedirs(outdir, exist_ok=True)

    # Build the full kwargs list for every run.
    run_kwargs_list = []
    for point in deduped:
        kwargs = dict(base_md_kwargs)
        kwargs.update(point)
        run_kwargs_list.append(kwargs)

    n_runs = len(run_kwargs_list)

    def _run_one(i, kwargs):
        run_dir = os.path.join(outdir, 'run_%04d' % i) if save else None

        # Resume: only when save_traj=True and traj.h5 already exists.
        # Checking params.yaml was wrong -- it is written for EVERY run, so
        # an existing params.yaml just means the run was started, not finished.
        # When save_traj=False there is no result file on disk; always re-run.
        if run_dir is not None:
            traj_path    = os.path.join(run_dir, 'traj.h5')
            already_done = save_traj and os.path.isfile(traj_path)
            if already_done:
                warnings.warn(
                    "run_%04d: traj.h5 already exists, skipping (resume). "
                    "Delete %s to re-run." % (i, run_dir),
                    UserWarning
                )
                from drift.slides_io import load_trajectory
                traj, _ = load_trajectory(traj_path)
                result = post_fn(traj, kwargs) if post_fn is not None else traj
                return {'params': kwargs, 'result': result, 'run_dir': run_dir}

        _log.info("sweep_md: starting run %04d / %04d", i, n_runs - 1)
        traj   = run_md(pos, calc_en_f, en_params, **kwargs)
        result = post_fn(traj, kwargs) if post_fn is not None else traj
        if run_dir is not None:
            _save_run(run_dir, traj, kwargs, save_traj=save_traj)
        return {'params': kwargs, 'result': result, 'run_dir': run_dir}

    # Progress bar: tqdm if available, plain print otherwise.
    # tqdm writes to stderr so it shows up in batch logs without polluting
    # stdout; the plain fallback uses stdout with flush for the same reason.
    def _make_pbar():
        if not verbose:
            return None
        try:
            import sys
            from tqdm import tqdm
            return tqdm(total=n_runs, desc='sweep_md', unit='run',
                        dynamic_ncols=True, file=sys.stderr)
        except ImportError:
            return None

    def _tick(pbar, done):
        if not verbose:
            return
        if pbar is not None:
            pbar.update(1)
        else:
            import sys
            print('[sweep_md] %d / %d' % (done, n_runs), flush=True,
                  file=sys.stderr)

    if n_jobs == 1:
        pbar = _make_pbar()
        results = []
        for i, kw in enumerate(run_kwargs_list):
            results.append(_run_one(i, kw))
            _tick(pbar, i + 1)
        if pbar is not None:
            pbar.close()
        return results

    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "joblib is required for n_jobs != 1. "
            "Install with: pip install joblib"
        )

    pbar = _make_pbar()
    try:
        # return_as='generator' (joblib >= 1.2) streams results as workers
        # finish, enabling per-completion progress updates.
        gen = Parallel(n_jobs=n_jobs, backend=backend,
                       return_as='generator')(
            delayed(_run_one)(i, kw) for i, kw in enumerate(run_kwargs_list)
        )
        results = []
        for result in gen:
            results.append(result)
            _tick(pbar, len(results))
    except TypeError:
        # Older joblib: collect all results at once, no per-completion progress.
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_run_one)(i, kw) for i, kw in enumerate(run_kwargs_list)
        )
        if pbar is not None:
            pbar.update(n_runs)
    finally:
        if pbar is not None:
            pbar.close()
    return results
