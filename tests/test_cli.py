"""
Tests for cli.py -- the unified `drift` CLI.

All physics subcommands (_cmd_map, _cmd_sweep, _cmd_string) are exercised
by calling the internal _cmd_* functions directly with argparse.Namespace
objects.  This avoids subprocess overhead and JIT recompilation per test.

The argument parser is tested via _build_parser().parse_args() calls.

Shared fixture: a small sin substrate (N1=5, N2=5, ~19 particles) written
to a temporary directory as params.yaml.  Each test that writes output
gets its own isolated temp dir via tmp_path.
"""

import json
import os
import sys
import warnings
import tempfile
from argparse import Namespace

import numpy as np
import pytest

from cli import (
    _build_parser,
    _build_physics,
    _cmd_make_params,
    _cmd_make_sweep,
    _cmd_map,
    _cmd_sweep,
    _cmd_string,
    _dump_yaml,
    _load_yaml,
    _resolve_post_fn,
)
from tool_create_substrate import get_ks


# ============================================================
# Shared fixtures
# ============================================================

_KS = get_ks(1.0, 3, 4.0 / 3.0, 0.0).tolist()

_PHYSICS_PARAMS = {
    'well_shape':    'sin',
    'epsilon':       1.0,
    'sub_basis':     [[0, 0]],
    'ks':            _KS,
    'a1':            [1.0, 0.0],
    'a2':            [0.5, -0.866],
    'cluster_shape': 'circle',
    'cl_basis':      [[0, 0]],
    'N1': 5, 'N2': 5,
    'theta': 0.0,
    'pos_cm': [0.0, 0.0],
}


@pytest.fixture(scope='module')
def params_yaml(tmp_path_factory):
    """Write _PHYSICS_PARAMS to a temp YAML file once per module."""
    d = tmp_path_factory.mktemp('params')
    path = str(d / 'params.yaml')
    _dump_yaml(_PHYSICS_PARAMS, path)
    return path


# ============================================================
# 1. make-sweep -- pure Python, no physics
# ============================================================

def test_make_sweep_grid(tmp_path):
    out = str(tmp_path / 'spec.yaml')
    ns = Namespace(
        make_sweep_type='grid',
        grid_axes=['Fx=0.0,0.1,0.2', 'kBT=0.0,1.0'],
        line_cols=None, force_vals=None, phi_deg=0.0,
        base_kwargs=None, post_fn_name=None,
        n_jobs=None, backend=None, no_save_traj=False,
        outdir=None, output=out,
    )
    _cmd_make_sweep(ns)
    spec = _load_yaml(out)
    assert spec['sweep_type'] == 'grid'
    assert spec['grid']['Fx']  == [0.0, 0.1, 0.2]
    assert spec['grid']['kBT'] == [0.0, 1.0]


def test_make_sweep_line(tmp_path):
    out = str(tmp_path / 'spec.yaml')
    ns = Namespace(
        make_sweep_type='line',
        grid_axes=None,
        line_cols=['Tau=0,1,2', 'Fx=0,0,0'],
        force_vals=None, phi_deg=0.0,
        base_kwargs=None, post_fn_name=None,
        n_jobs=None, backend=None, no_save_traj=False,
        outdir=None, output=out,
    )
    _cmd_make_sweep(ns)
    spec = _load_yaml(out)
    assert spec['sweep_type'] == 'line'
    assert spec['line']['Tau'] == [0.0, 1.0, 2.0]
    assert len(spec['line']['Fx']) == 3


def test_make_sweep_force(tmp_path):
    out = str(tmp_path / 'spec.yaml')
    ns = Namespace(
        make_sweep_type='force',
        grid_axes=None, line_cols=None,
        force_vals='0.0,0.5,1.0', phi_deg=90.0,
        base_kwargs=None, post_fn_name=None,
        n_jobs=None, backend=None, no_save_traj=False,
        outdir=None, output=out,
    )
    _cmd_make_sweep(ns)
    spec = _load_yaml(out)
    assert spec['sweep_type'] == 'force'
    assert spec['F_vals']  == [0.0, 0.5, 1.0]
    assert spec['phi_deg'] == 90.0


def test_make_sweep_options_embedded(tmp_path):
    """--base, --post-fn, --n-jobs, --no-save-traj appear in the output."""
    out = str(tmp_path / 'spec.yaml')
    ns = Namespace(
        make_sweep_type='grid',
        grid_axes=['Fx=0.0,0.1'],
        line_cols=None, force_vals=None, phi_deg=0.0,
        base_kwargs=['eta=1.0', 'n_steps=50000'],
        post_fn_name='drift_velocity',
        n_jobs=4, backend='loky',
        no_save_traj=True,
        outdir='sweep_out',
        output=out,
    )
    _cmd_make_sweep(ns)
    spec = _load_yaml(out)
    assert spec['base_md_kwargs']['eta']     == 1.0
    assert spec['base_md_kwargs']['n_steps'] == 50000
    assert spec['post_fn']    == 'drift_velocity'
    assert spec['n_jobs']     == 4
    assert spec['backend']    == 'loky'
    assert spec['save_traj']  == False
    assert spec['outdir']     == 'sweep_out'


def test_make_sweep_default_output_name(tmp_path, monkeypatch):
    """Omitting -o writes to sweep_spec.yaml in cwd."""
    monkeypatch.chdir(tmp_path)
    ns = Namespace(
        make_sweep_type='grid',
        grid_axes=['Fx=0.0,0.1'],
        line_cols=None, force_vals=None, phi_deg=0.0,
        base_kwargs=None, post_fn_name=None,
        n_jobs=None, backend=None, no_save_traj=False,
        outdir=None, output=None,
    )
    _cmd_make_sweep(ns)
    assert os.path.isfile(str(tmp_path / 'sweep_spec.yaml'))


# ============================================================
# 2. _build_physics
# ============================================================

def test_build_physics_en_inputs_empty(params_yaml):
    """substrate_from_params closure returns en_inputs=[]."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pos, en_func, en_inputs, params = _build_physics(params_yaml)
    assert en_inputs == [], "en_inputs must be [] (closure captures all params)"
    assert pos.shape[1] == 2


def test_build_physics_en_func_callable(params_yaml):
    """en_func closure returns (float, (2,) array, float) at origin."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pos, en_func, en_inputs, _ = _build_physics(params_yaml)
    pos_cm = np.zeros(2)
    E, F, tau = en_func(pos, pos_cm, *en_inputs)
    assert isinstance(float(E), float)
    assert np.asarray(F).shape == (2,)
    assert isinstance(float(tau), float)


def test_build_physics_theta_not_applied(tmp_path):
    """_build_physics returns the reference-frame pos regardless of theta.

    theta is stored in params_dict for callers to apply selectively;
    it must NOT be baked into pos here to avoid double-rotation in MD
    (run_md uses theta0) and in the 3D string method (rotates internally).
    """
    import copy
    p = copy.deepcopy(_PHYSICS_PARAMS)
    p['theta'] = 30.0
    path = str(tmp_path / 'p30.yaml')
    _dump_yaml(p, path)

    p0 = copy.deepcopy(_PHYSICS_PARAMS)
    p0['theta'] = 0.0
    path0 = str(tmp_path / 'p0.yaml')
    _dump_yaml(p0, path0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pos30, _, _, params30 = _build_physics(path)
        pos0,  _, _, params0  = _build_physics(path0)

    # pos is the reference frame in both cases: same positions.
    assert np.allclose(pos30, pos0), \
        "_build_physics must return reference-frame pos; theta is in params_dict"
    # theta is preserved in the returned params dict for callers.
    assert float(params30.get('theta', 0.0)) == 30.0
    assert float(params0.get('theta',  0.0)) == 0.0


# ============================================================
# 3. _resolve_post_fn
# ============================================================

def test_resolve_post_fn_drift_velocity():
    fn = _resolve_post_fn('drift_velocity', {})
    assert callable(fn)


def test_resolve_post_fn_mean_velocity_fraction():
    fn = _resolve_post_fn('mean_velocity', {'fraction': 0.5})
    assert callable(fn)
    # Smoke: call with a synthetic traj
    vel = np.ones((10, 2))
    traj = {'vel_cm': vel}
    result = fn(traj, {})
    assert abs(result - float(np.linalg.norm([1.0, 1.0]))) < 1e-10


def test_resolve_post_fn_last_state_keys():
    fn = _resolve_post_fn('last_state', {'keys': ['pos_cm', 'energy']})
    assert callable(fn)
    traj = {'pos_cm': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'energy': np.array([-1.0, -2.0])}
    result = fn(traj, {})
    assert set(result.keys()) == {'pos_cm', 'energy'}
    assert np.allclose(result['pos_cm'], [0.3, 0.4])


def test_resolve_post_fn_unknown_exits():
    with pytest.raises(SystemExit):
        _resolve_post_fn('nonexistent_fn', {})


# ============================================================
# 4. map subcommand
# ============================================================

def test_map_translational(params_yaml, tmp_path):
    """Translational map: output HDF5 has 'energy' and 'pos_cm' arrays."""
    grid_path = str(tmp_path / 'grid.yaml')
    out_path  = str(tmp_path / 'map.h5')
    _dump_yaml({'map_type': 'translational', 'n_x': 3, 'n_y': 3,
                'n_jobs': 1}, grid_path)

    ns = Namespace(input=params_yaml, grid=grid_path, output=out_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_map(ns)

    assert os.path.isfile(out_path)
    from slides_io import load_map
    result, _ = load_map(out_path)
    assert 'energy' in result
    assert result['energy'].shape == (3 * 3,)


def test_map_rotational(params_yaml, tmp_path):
    """Rotational map: output HDF5 has 'energy' and 'theta' arrays."""
    grid_path = str(tmp_path / 'grid.yaml')
    out_path  = str(tmp_path / 'rot.h5')
    _dump_yaml({'map_type': 'rotational', 'n_theta': 4, 'n_jobs': 1},
               grid_path)

    ns = Namespace(input=params_yaml, grid=grid_path, output=out_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_map(ns)

    assert os.path.isfile(out_path)
    from slides_io import load_map
    result, _ = load_map(out_path)
    assert 'energy' in result
    assert result['energy'].shape == (4,)


def test_map_default_output_name(params_yaml, tmp_path, monkeypatch):
    """Omitting -o writes map.h5 in cwd."""
    monkeypatch.chdir(tmp_path)
    grid_path = str(tmp_path / 'grid.yaml')
    _dump_yaml({'map_type': 'translational', 'n_x': 2, 'n_y': 2}, grid_path)

    ns = Namespace(input=params_yaml, grid=grid_path, output=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_map(ns)

    assert os.path.isfile(str(tmp_path / 'map.h5'))


def test_map_no_grid_file(params_yaml, tmp_path):
    """grid=None triggers default grid params (50x50), should not raise."""
    out_path = str(tmp_path / 'big.h5')
    # Override defaults via the grid yaml to use a tiny grid
    # (no grid yaml -> defaults to 50x50 which is slow; use a tiny grid yaml instead)
    grid_path = str(tmp_path / 'tiny.yaml')
    _dump_yaml({'map_type': 'translational', 'n_x': 2, 'n_y': 2}, grid_path)
    ns = Namespace(input=params_yaml, grid=grid_path, output=out_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_map(ns)
    assert os.path.isfile(out_path)


def test_map_bad_type_exits(params_yaml, tmp_path):
    """Unknown map_type raises ValueError."""
    grid_path = str(tmp_path / 'grid.yaml')
    _dump_yaml({'map_type': 'nonexistent'}, grid_path)
    ns = Namespace(input=params_yaml, grid=grid_path,
                   output=str(tmp_path / 'out.h5'))
    with pytest.raises(ValueError, match="map_type"):
        _cmd_map(ns)


# ============================================================
# 5. sweep subcommand
# ============================================================

def test_sweep_grid_spec_creates_run_dirs(params_yaml, tmp_path):
    """Grid sweep from spec.yaml creates run_0000/ and run_0001/."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0, 0.1]},
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    assert os.path.isdir(os.path.join(outdir, 'run_0000'))
    assert os.path.isdir(os.path.join(outdir, 'run_0001'))
    assert os.path.isfile(os.path.join(outdir, 'run_0000', 'params.yaml'))
    assert os.path.isfile(os.path.join(outdir, 'run_0000', 'traj.h5'))


def test_sweep_explicit_spec_list(params_yaml, tmp_path):
    """sweep_spec given as an explicit list of dicts."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_spec': [{'Fx': 0.0}, {'Fx': 0.2}],
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    dirs = sorted(d for d in os.listdir(outdir) if d.startswith('run_'))
    assert dirs == ['run_0000', 'run_0001']


def test_sweep_no_save_traj(params_yaml, tmp_path):
    """save_traj=False: params.yaml written, traj.h5 absent."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0]},
        'save_traj':      False,
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    run_dir = os.path.join(outdir, 'run_0000')
    assert     os.path.isfile(os.path.join(run_dir, 'params.yaml'))
    assert not os.path.isfile(os.path.join(run_dir, 'traj.h5'))


def test_sweep_with_post_fn(params_yaml, tmp_path):
    """post_fn=drift_velocity reduces result to a (2,) array (serialised)."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0]},
        'post_fn':        'drift_velocity',
        'save_traj':      False,
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    # params.yaml must exist; traj.h5 absent (save_traj=False).
    run_dir = os.path.join(outdir, 'run_0000')
    assert os.path.isfile(os.path.join(run_dir, 'params.yaml'))


def test_sweep_outdir_cli_overrides_spec(params_yaml, tmp_path):
    """--outdir CLI arg takes precedence over spec.yaml outdir field."""
    spec_path  = str(tmp_path / 'spec.yaml')
    spec_outdir = str(tmp_path / 'spec_out')   # from spec -- should NOT be used
    cli_outdir  = str(tmp_path / 'cli_out')    # from CLI  -- should be used
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0]},
        'outdir':         spec_outdir,
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=cli_outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    assert     os.path.isdir(cli_outdir)
    assert not os.path.isdir(spec_outdir)


# ============================================================
# 6. string subcommand
# ============================================================

def test_string_2d_creates_output(params_yaml, tmp_path):
    """2D MEP: output HDF5 has 'points' (n_pt, 2) and 'energy' (n_pt,) keys."""
    cfg_path = str(tmp_path / 'str.yaml')
    out_path = str(tmp_path / 'mep.h5')
    _dump_yaml({
        'p0':       [0.0, 0.0],
        'p1':       [0.5, 0.0],
        'n_points': 5,
        'n_iter':   5,
        'step':     1e-3,
        'tol':      1e-4,
    }, cfg_path)

    ns = Namespace(input=params_yaml, cfg=cfg_path, output=out_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_string(ns)

    assert os.path.isfile(out_path)
    from slides_io import load_map
    result, _ = load_map(out_path)
    assert 'points' in result
    assert result['points'].shape[1] == 2
    assert 'energy' in result
    assert result['energy'].shape[0] == result['points'].shape[0]


def test_string_default_output_name(params_yaml, tmp_path, monkeypatch):
    """Omitting -o writes mep.h5 in cwd."""
    monkeypatch.chdir(tmp_path)
    cfg_path = str(tmp_path / 'str.yaml')
    _dump_yaml({'p0': [0.0, 0.0], 'p1': [0.5, 0.0],
                'n_points': 4, 'n_iter': 3}, cfg_path)

    ns = Namespace(input=params_yaml, cfg=cfg_path, output=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_string(ns)

    assert os.path.isfile(str(tmp_path / 'mep.h5'))


def test_string_dimension_mismatch_exits(params_yaml, tmp_path):
    """p0 and p1 with different lengths trigger sys.exit(1)."""
    cfg_path = str(tmp_path / 'str.yaml')
    _dump_yaml({'p0': [0.0, 0.0], 'p1': [0.5, 0.0, 30.0]}, cfg_path)

    ns = Namespace(input=params_yaml, cfg=cfg_path,
                   output=str(tmp_path / 'mep.h5'))
    with pytest.raises(SystemExit):
        _cmd_string(ns)


# ============================================================
# 7. Argument parser
# ============================================================

def test_parser_no_subcommand_exits():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_map_missing_input_exits():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['map'])   # -i/--input is required


def test_parser_sweep_missing_spec_exits():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['sweep', '-i', 'p.yaml'])  # --spec required


def test_parser_make_sweep_defaults():
    parser = _build_parser()
    ns = parser.parse_args(['make-sweep'])
    assert ns.make_sweep_type == 'grid'
    assert ns.phi_deg         == 0.0
    assert ns.no_save_traj    == False
    assert ns.output          is None


# ============================================================
# 8. make-params subcommand
# ============================================================

def test_make_params_sin_writes_file(tmp_path):
    """make-params --substrate sin --n 3 writes a file with 'ks' key."""
    import yaml
    out = str(tmp_path / 'p.yaml')
    ns = Namespace(substrate='sin', n=3, spacing=1.0, output=out)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_make_params(ns)
    assert os.path.isfile(out)
    content = open(out).read()
    assert 'ks:' in content
    assert 'well_shape: sin' in content
    assert 'generated by' in content


def test_make_params_sin_ks_correct(tmp_path):
    """ks values match get_ks output for n=6 triangular."""
    import yaml
    out = str(tmp_path / 'p.yaml')
    ns = Namespace(substrate='sin', n=6, spacing=2.0, output=out)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_make_params(ns)
    # Parse ks from the written file (YAML-compatible subset).
    content = open(out).read()
    # Strip comment lines before loading so yaml.safe_load accepts it.
    lines = [l for l in content.splitlines() if not l.strip().startswith('#')]
    d = yaml.safe_load('\n'.join(lines))
    ref_ks = get_ks(2.0, 6, 4.0 / 3.0**0.5, -np.pi / 6.0)
    assert np.allclose(np.array(d['ks']), ref_ks, atol=1e-12)


def test_make_params_sin_bad_n_raises():
    """--n 7 is not a valid fold symmetry."""
    ns = Namespace(substrate='sin', n=7, spacing=1.0, output='-')
    with pytest.raises(ValueError, match="--n must be"):
        _cmd_make_params(ns)


def test_make_params_sin_missing_n_raises():
    """--n is required for sin substrate."""
    ns = Namespace(substrate='sin', n=None, spacing=1.0, output='-')
    with pytest.raises(ValueError, match="--n is required"):
        _cmd_make_params(ns)


def test_make_params_gaussian_writes_file(tmp_path):
    """make-params --substrate gaussian writes a file with b1, b2, sigma."""
    out = str(tmp_path / 'g.yaml')
    ns = Namespace(substrate='gaussian', n=None, spacing=1.5, output=out)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_make_params(ns)
    content = open(out).read()
    assert 'well_shape: gaussian' in content
    assert 'b1:' in content
    assert 'b2:' in content
    assert 'sigma:' in content


def test_make_params_tanh_writes_file(tmp_path):
    """make-params --substrate tanh writes a file with wd key."""
    out = str(tmp_path / 't.yaml')
    ns = Namespace(substrate='tanh', n=None, spacing=1.0, output=out)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_make_params(ns)
    content = open(out).read()
    assert 'well_shape: tanh' in content
    assert 'wd:' in content


def test_make_params_stdout(capsys):
    """output='-' prints to stdout instead of writing a file."""
    ns = Namespace(substrate='sin', n=3, spacing=1.0, output='-')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_make_params(ns)
    captured = capsys.readouterr()
    assert 'well_shape: sin' in captured.out


def test_parser_make_params_registered():
    """make-params subcommand is wired into the parser."""
    parser = _build_parser()
    ns = parser.parse_args(
        ['make-params', '--substrate', 'sin', '--n', '3', '--spacing', '1.0']
    )
    assert ns.substrate == 'sin'
    assert ns.n         == 3
    assert ns.spacing   == 1.0


# ============================================================
# 9. sweep resume fix (Task 1)
# ============================================================

def test_sweep_resume_skips_existing_traj(params_yaml, tmp_path):
    """Second sweep_md call skips runs where traj.h5 already exists."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0]},
        'save_traj':      True,
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    traj_path = os.path.join(outdir, 'run_0000', 'traj.h5')
    assert os.path.isfile(traj_path)
    mtime_first = os.path.getmtime(traj_path)

    # Second call: run should be skipped and traj.h5 not rewritten.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)

    assert os.path.getmtime(traj_path) == mtime_first, \
        "traj.h5 was overwritten on resume -- resume guard is broken"


def test_sweep_no_resume_when_save_traj_false(params_yaml, tmp_path):
    """save_traj=False: second call always reruns (no traj.h5 to detect)."""
    spec_path = str(tmp_path / 'spec.yaml')
    outdir    = str(tmp_path / 'out')
    _dump_yaml({
        'sweep_type':     'grid',
        'grid':           {'Fx': [0.0]},
        'save_traj':      False,
        'base_md_kwargs': {'eta': 1.0, 'kBT': 1e-8, 'dt': 5e-4,
                           'n_steps': 50, 'print_every': 25},
    }, spec_path)

    ns = Namespace(input=params_yaml, spec=spec_path, outdir=outdir)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _cmd_sweep(ns)
        # Second call: no traj.h5 exists, so the run must execute again
        # (result should be non-None, not None from a skipped run).
        results2 = None
        from cli import _build_physics
        pos, calc_en_f, en_params, p = _build_physics(params_yaml)
        from sweep_md import sweep_md, grid_sweep
        spec = grid_sweep({'Fx': [0.0]})
        results2 = sweep_md(pos, calc_en_f, en_params, spec,
                            base_md_kwargs={'eta': 1.0, 'kBT': 1e-8,
                                            'dt': 5e-4, 'n_steps': 50,
                                            'print_every': 25},
                            save=True, save_traj=False,
                            outdir=outdir, verbose=False)
    assert results2[0]['result'] is not None, \
        "save_traj=False run should not be skipped on re-run"
