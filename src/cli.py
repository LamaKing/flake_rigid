"""
Unified command-line interface for the DRIFT package.

Usage
-----
    drift map    -i params.yaml --grid  grid.yaml   -o output.h5
    drift sweep  -i params.yaml --spec  spec.yaml   --outdir sweep_out/
    drift string -i params.yaml --cfg   string.yaml -o path.h5
    drift make-sweep [options]  -o spec.yaml

Physics input (params.yaml)
---------------------------
A single YAML file describing both the substrate and the cluster:

    # substrate
    well_shape:    sin            # 'sin', 'gaussian', or 'tanh'
    epsilon:       1.0
    sub_basis:     [[0, 0]]
    ks:            [...]          # from get_ks; required for 'sin'
    a1:            [1.0, 0.0]     # required for 'gaussian'/'tanh'
    a2:            [0.5, 0.866]   # required for 'gaussian'/'tanh'
    # cluster
    cluster_shape: circle
    N1: 9
    N2: 9
    cl_basis:      [[0, 0]]
    theta:         0.0            # optional initial rotation (degrees)
    pos_cm:        [0.0, 0.0]     # ignored at CLI level

Units and conventions follow tool_create_substrate and tool_create_cluster.
Angles are in degrees at the user level; the string method converts internally.
"""

import argparse
import os
import sys
import logging

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())


# ============================================================
# YAML / I/O helpers
# ============================================================

def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _dump_yaml(obj, path):
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, 'w') as fh:
        yaml.dump(obj, fh, default_flow_style=False, sort_keys=False)


def _build_physics(params_path):
    """Load params.yaml and return (pos, calc_en_f, en_params, params_dict).

    Imports are deferred so that the CLI starts fast even on cold Numba.
    """
    from tool_create_substrate import substrate_from_params
    from tool_create_cluster import cluster_from_params, rotate

    params = _load_yaml(params_path)

    _, calc_en_f, en_params = substrate_from_params(params)
    pos = cluster_from_params(params)

    # Apply initial rotation if specified in params.
    theta = float(params.get('theta', 0.0))
    if theta != 0.0:
        pos = rotate(pos, theta)

    return pos, calc_en_f, en_params, params


# ============================================================
# map subcommand
# ============================================================

_MAP_TYPES = ('translational', 'rotational', 'rototrasl')


def _cmd_map(args):
    """Compute a static energy map and write the result to HDF5.

    grid.yaml keys (all optional if defaults are acceptable):
        map_type:  'translational' | 'rotational' | 'rototrasl'
                   (default: 'translational')
        n_x, n_y:  int -- grid size along x and y (default: 50)
        n_theta:   int -- number of orientation angles (rototrasl only)
        frac_x:    [lo, hi] -- fractional x range (default: [0, 1])
        frac_y:    [lo, hi] -- fractional y range (default: [0, 1])
        theta_range: [lo, hi] -- in degrees (default: [0, 360])
        n_jobs:    int -- joblib workers (default: 1)

    For 'sin' substrates u_inv is taken as eye(2); for 'gaussian'/'tanh'
    it is derived from the b1/b2 lattice vectors in params.yaml.
    """
    import numpy as np
    from maps import translational_map, rotational_map, rototrasl_map
    from slides_io import save_map

    pos, calc_en_f, en_params, params = _build_physics(args.input)

    grid = _load_yaml(args.grid) if args.grid else {}

    map_type   = grid.get('map_type', 'translational')
    n_x        = int(grid.get('n_x', 50))
    n_y        = int(grid.get('n_y', 50))
    n_theta    = int(grid.get('n_theta', 36))
    frac_x     = tuple(grid.get('frac_x', [0.0, 1.0]))
    frac_y     = tuple(grid.get('frac_y', [0.0, 1.0]))
    theta_range = tuple(grid.get('theta_range', [0.0, 360.0]))
    n_jobs     = int(grid.get('n_jobs', 1))

    if map_type not in _MAP_TYPES:
        print("ERROR: map_type '%s' not in %s" % (map_type, _MAP_TYPES),
              file=sys.stderr)
        sys.exit(1)

    # u_inv: metric matrix for real-space grid.  sin substrates use identity
    # because the plane-wave potential is Cartesian by construction.
    well_shape = params.get('well_shape', 'sin')
    if well_shape in ('gaussian', 'tanh'):
        from tool_create_substrate import calc_matrices_bvect
        b1 = params['b1']
        b2 = params['b2']
        _, u_inv = calc_matrices_bvect(b1, b2)
    else:
        u_inv = np.eye(2)

    print("drift map: type=%s  grid=(%d x %d)  n_jobs=%d"
          % (map_type, n_x, n_y, n_jobs), flush=True)

    if map_type == 'translational':
        result = translational_map(pos, calc_en_f, en_params, u_inv,
                                   n_x, n_y,
                                   frac_x=frac_x, frac_y=frac_y,
                                   n_jobs=n_jobs)
    elif map_type == 'rotational':
        result = rotational_map(pos, calc_en_f, en_params,
                                n_theta=n_theta,
                                theta_range=theta_range,
                                n_jobs=n_jobs)
    else:  # rototrasl
        result = rototrasl_map(pos, calc_en_f, en_params, u_inv,
                               n_x, n_y, n_theta,
                               frac_x=frac_x, frac_y=frac_y,
                               theta_range=theta_range,
                               n_jobs=n_jobs)

    out = args.output if args.output else 'map.h5'
    save_map(result, out, params=params)
    print("drift map: saved to %s" % out, flush=True)


# ============================================================
# sweep subcommand
# ============================================================

_POST_FN_REGISTRY = {
    'last_state':     'last_state',
    'mean_velocity':  'mean_velocity',
    'drift_velocity': 'drift_velocity',
}


def _resolve_post_fn(name, spec_params):
    """Resolve a post_fn by name string, using optional spec_params for kwargs.

    spec_params keys used per post_fn:
        mean_velocity:  fraction (float, default 0.2)
        last_state:     keys    (list, default None)
    """
    from sweep_md import last_state, mean_velocity, drift_velocity

    if name == 'drift_velocity':
        return drift_velocity()
    if name == 'mean_velocity':
        fraction = float(spec_params.get('fraction', 0.2))
        return mean_velocity(fraction=fraction)
    if name == 'last_state':
        keys = spec_params.get('keys', None)
        return last_state(keys=keys)

    print("ERROR: unknown post_fn '%s'. Choose from: %s"
          % (name, sorted(_POST_FN_REGISTRY)), file=sys.stderr)
    sys.exit(1)


def _cmd_sweep(args):
    """Run sweep_md from a spec YAML.

    spec.yaml keys:
        sweep_spec:     list of dicts (run_md kwargs overrides), OR...
        sweep_type:     'grid' | 'line' | 'force'  -- build spec automatically
            grid:       dict key -> list of values  (for sweep_type='grid')
            line:       dict key -> list of values  (for sweep_type='line')
            F_vals:     list of floats              (for sweep_type='force')
            phi_deg:    float                       (for sweep_type='force')
        base_md_kwargs: dict of default run_md kwargs
        post_fn:        str  -- name from drift_velocity/mean_velocity/last_state
        post_fn_params: dict -- kwargs forwarded to the post_fn factory
        n_jobs:         int  (default 1)
        backend:        str  (default 'loky')
        save_traj:      bool (default True)
        outdir:         str  (default '.')
    """
    from sweep_md import sweep_md, grid_sweep, line_sweep, force_sweep

    pos, calc_en_f, en_params, _ = _build_physics(args.input)

    spec_dict = _load_yaml(args.spec)

    # Build sweep_spec.
    if 'sweep_spec' in spec_dict:
        sweep_spec = spec_dict['sweep_spec']
    else:
        sweep_type = spec_dict.get('sweep_type', 'grid')
        if sweep_type == 'grid':
            sweep_spec = grid_sweep(spec_dict['grid'])
        elif sweep_type == 'line':
            sweep_spec = line_sweep(spec_dict['line'])
        elif sweep_type == 'force':
            sweep_spec = force_sweep(spec_dict['F_vals'],
                                     phi_deg=float(spec_dict.get('phi_deg', 0.0)))
        else:
            print("ERROR: sweep_type '%s' not in (grid, line, force)"
                  % sweep_type, file=sys.stderr)
            sys.exit(1)

    base_md_kwargs = spec_dict.get('base_md_kwargs', {})
    n_jobs         = int(spec_dict.get('n_jobs', 1))
    backend        = str(spec_dict.get('backend', 'loky'))
    save_traj      = bool(spec_dict.get('save_traj', True))
    outdir         = args.outdir if args.outdir else spec_dict.get('outdir', '.')

    post_fn = None
    if 'post_fn' in spec_dict:
        post_fn = _resolve_post_fn(spec_dict['post_fn'],
                                   spec_dict.get('post_fn_params', {}))

    print("drift sweep: %d points  n_jobs=%d  backend=%s  outdir=%s"
          % (len(sweep_spec), n_jobs, backend, outdir), flush=True)

    results = sweep_md(pos, calc_en_f, en_params, sweep_spec,
                       base_md_kwargs=base_md_kwargs,
                       post_fn=post_fn,
                       n_jobs=n_jobs, backend=backend,
                       outdir=outdir, save=True, save_traj=save_traj,
                       verbose=True)

    print("drift sweep: done. %d runs written to %s" % (len(results), outdir),
          flush=True)


# ============================================================
# string subcommand
# ============================================================

def _cmd_string(args):
    """Run find_mep from a string config YAML and save the path to HDF5.

    string.yaml keys:
        p0:          list of 2 or 3 floats -- start point [x, y] or [x, y, theta_deg]
        p1:          list of 2 or 3 floats -- end   point [x, y] or [x, y, theta_deg]
        n_points:    int   -- number of string beads (default 20)
        n_iter:      int   -- number of string iterations (default 200)
        step:        float -- gradient step size (default 1e-3)
        tol:         float -- convergence tolerance (default 1e-6)
        scale:       list of floats -- arc-length rescaling factors
                     (default [1, 1] for 2D; [1, 1, 1] for 3D)

    For 3D (len(p0)==3) pos is kept in the reference frame (theta=0); the
    string method rotates internally.  For 2D (len(p0)==2) pos should already
    be rotated to the desired orientation.
    """
    import numpy as np
    from string_method import find_mep
    from slides_io import save_map

    pos, calc_en_f, en_params, params = _build_physics(args.input)

    cfg = _load_yaml(args.cfg)

    p0 = cfg['p0']
    p1 = cfg['p1']

    if len(p0) != len(p1):
        print("ERROR: p0 and p1 must have the same dimension.", file=sys.stderr)
        sys.exit(1)

    dim      = len(p0)
    n_points = int(cfg.get('n_points', 20))
    n_iter   = int(cfg.get('n_iter', 200))
    step     = float(cfg.get('step', 1e-3))
    tol      = float(cfg.get('tol', 1e-6))
    scale    = cfg.get('scale', [1.0] * dim)

    print("drift string: dim=%d  n_points=%d  n_iter=%d"
          % (dim, n_points, n_iter), flush=True)

    result = find_mep(pos, calc_en_f, en_params,
                      p0=p0, p1=p1,
                      n_points=n_points, n_iter=n_iter,
                      step=step, tol=tol,
                      scale=scale)

    out = args.output if args.output else 'mep.h5'
    save_map(result, out, params=params)
    print("drift string: saved to %s" % out, flush=True)


# ============================================================
# make-sweep subcommand
# ============================================================

def _cmd_make_sweep(args):
    """Generate a sweep_spec.yaml without running any physics.

    Options:
        --type grid | line | force
        --grid KEY=V1,V2,...  (repeatable, for --type grid)
        --line KEY=V1,V2,...  (repeatable, for --type line)
        --force-vals V1,V2,... (for --type force)
        --phi-deg FLOAT        (for --type force, default 0)
        --base KEY=VAL         (repeatable, added to base_md_kwargs)
        --post-fn NAME         (appended to spec yaml as post_fn)
        --n-jobs INT
        --backend STR
        --no-save-traj
        -o / --output PATH     (default: sweep_spec.yaml)
    """
    import ast

    sweep_type = args.make_sweep_type

    def _parse_kvlist(items):
        """Parse KEY=V1,V2,... -> {KEY: [float(V1), float(V2), ...]}."""
        out = {}
        for item in (items or []):
            k, _, vs = item.partition('=')
            out[k.strip()] = [float(v) for v in vs.split(',')]
        return out

    def _parse_kvscalar(items):
        """Parse KEY=VAL -> {KEY: <best type>}."""
        out = {}
        for item in (items or []):
            k, _, v = item.partition('=')
            try:
                out[k.strip()] = ast.literal_eval(v.strip())
            except (ValueError, SyntaxError):
                out[k.strip()] = v.strip()
        return out

    if sweep_type == 'grid':
        grid_axes  = _parse_kvlist(args.grid_axes)
        spec_block = {'sweep_type': 'grid', 'grid': grid_axes}
    elif sweep_type == 'line':
        line_cols  = _parse_kvlist(args.line_cols)
        spec_block = {'sweep_type': 'line', 'line': line_cols}
    elif sweep_type == 'force':
        F_vals     = [float(v) for v in (args.force_vals or '').split(',') if v]
        phi_deg    = float(args.phi_deg or 0.0)
        spec_block = {'sweep_type': 'force', 'F_vals': F_vals, 'phi_deg': phi_deg}
    else:
        print("ERROR: --type must be grid, line, or force.", file=sys.stderr)
        sys.exit(1)

    base_md_kwargs = _parse_kvscalar(args.base_kwargs)
    if base_md_kwargs:
        spec_block['base_md_kwargs'] = base_md_kwargs

    if args.post_fn_name:
        spec_block['post_fn'] = args.post_fn_name
    if args.n_jobs:
        spec_block['n_jobs'] = int(args.n_jobs)
    if args.backend:
        spec_block['backend'] = args.backend
    if args.no_save_traj:
        spec_block['save_traj'] = False
    if args.outdir:
        spec_block['outdir'] = args.outdir

    out = args.output if args.output else 'sweep_spec.yaml'
    _dump_yaml(spec_block, out)
    print("drift make-sweep: wrote %s" % out, flush=True)


# ============================================================
# Argument parser
# ============================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        prog='drift',
        description='DRIFT: rigid-cluster statics and dynamics on a substrate.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable DEBUG logging.')
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- map ----
    p_map = sub.add_parser('map', help='Compute a static energy map.')
    p_map.add_argument('-i', '--input',  required=True, metavar='PARAMS',
                       help='Physics parameters YAML.')
    p_map.add_argument('--grid',         metavar='GRID_YAML',
                       help='Grid config YAML (defaults used if omitted).')
    p_map.add_argument('-o', '--output', metavar='OUT_H5',
                       help='Output HDF5 file (default: map.h5).')

    # ---- sweep ----
    p_sw = sub.add_parser('sweep', help='Run an MD parameter sweep.')
    p_sw.add_argument('-i', '--input',   required=True, metavar='PARAMS',
                      help='Physics parameters YAML.')
    p_sw.add_argument('--spec',          required=True, metavar='SPEC_YAML',
                      help='Sweep spec YAML (from make-sweep or hand-written).')
    p_sw.add_argument('--outdir',        metavar='DIR',
                      help='Output directory (overrides spec.yaml outdir).')

    # ---- string ----
    p_str = sub.add_parser('string', help='Find a minimum energy path.')
    p_str.add_argument('-i', '--input',  required=True, metavar='PARAMS',
                       help='Physics parameters YAML.')
    p_str.add_argument('--cfg',          required=True, metavar='STRING_YAML',
                       help='String method config YAML.')
    p_str.add_argument('-o', '--output', metavar='OUT_H5',
                       help='Output HDF5 file (default: mep.h5).')

    # ---- make-sweep ----
    p_ms = sub.add_parser('make-sweep',
                           help='Generate a sweep spec YAML (no physics run).')
    p_ms.add_argument('--type', dest='make_sweep_type',
                      choices=('grid', 'line', 'force'), default='grid',
                      help='Sweep layout type (default: grid).')
    p_ms.add_argument('--grid', dest='grid_axes', action='append',
                      metavar='KEY=V1,V2,...',
                      help='Grid axis. Repeatable. E.g. --grid Fx=0,0.1,0.2')
    p_ms.add_argument('--line', dest='line_cols', action='append',
                      metavar='KEY=V1,V2,...',
                      help='Line column. Repeatable.')
    p_ms.add_argument('--force-vals', metavar='V1,V2,...',
                      help='Comma-separated force magnitudes for --type force.')
    p_ms.add_argument('--phi-deg', type=float, default=0.0,
                      help='Force angle in degrees (default: 0).')
    p_ms.add_argument('--base', dest='base_kwargs', action='append',
                      metavar='KEY=VAL',
                      help='Base MD kwarg. Repeatable. E.g. --base eta=1.0')
    p_ms.add_argument('--post-fn', dest='post_fn_name',
                      choices=list(_POST_FN_REGISTRY),
                      help='post_fn name to embed in spec.')
    p_ms.add_argument('--n-jobs', type=int, metavar='N',
                      help='n_jobs to embed in spec.')
    p_ms.add_argument('--backend', metavar='STR',
                      help='joblib backend to embed in spec.')
    p_ms.add_argument('--no-save-traj', action='store_true',
                      help='Set save_traj=False in spec.')
    p_ms.add_argument('--outdir', metavar='DIR',
                      help='outdir to embed in spec.')
    p_ms.add_argument('-o', '--output', metavar='SPEC_YAML',
                      help='Output YAML file (default: sweep_spec.yaml).')

    return parser


# ============================================================
# Entry point
# ============================================================

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(levelname)s %(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(levelname)s: %(message)s')

    dispatch = {
        'map':        _cmd_map,
        'sweep':      _cmd_sweep,
        'string':     _cmd_string,
        'make-sweep': _cmd_make_sweep,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
