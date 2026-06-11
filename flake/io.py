"""
File I/O for rigid-cluster simulations.

HDF5 format (via h5py) is used for all numerical results.  Each dataset
name in the file matches the dict key returned by the map functions.
Metadata (scalar parameters, substrate/cluster settings) is stored as
HDF5 attributes; numpy arrays and nested structures in params are
serialised as JSON strings to avoid HDF5 group complexity.

YAML format (via PyYAML) is used for human-editable parameter files.

Neither h5py nor yaml is imported at module level -- both are guarded
so the module can be imported even if the optional dependency is absent;
only the relevant function call raises ImportError.
"""

import json
import numpy as np


# ============================================================
# HDF5 save/load
# ============================================================

def save_map(result_dict, filename, params=None):
    """Save a map result dict to HDF5.

    Args:
        result_dict: dict of numpy arrays (output of translational_map etc.).
        filename:    str -- output path (should end in .h5 or .hdf5).
        params:      optional dict of scalar/string/array metadata.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    with h5py.File(filename, 'w') as f:
        for key, val in result_dict.items():
            f.create_dataset(key, data=np.asarray(val, dtype=np.float64))
        if params is not None:
            _write_attrs(f, params)


def load_map(filename):
    """Load HDF5 map file.

    Returns:
        (result_dict, params_dict)  --  both plain Python dicts.
        result_dict values are numpy arrays; params_dict values are
        Python scalars, strings, or dicts (JSON-decoded where applicable).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    result_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            result_dict[key] = f[key][...]
        params_dict = _read_attrs(f)
    return result_dict, params_dict


def save_trajectory(traj_dict, filename, params=None):
    """Save an MD trajectory dict to HDF5.

    Expected keys: 't', 'energy', 'pos_cm', 'vel_cm', 'theta',
    'omega', 'force', 'torque'.  Any subset is accepted.

    Args:
        traj_dict: dict of numpy arrays.
        filename:  str -- output path.
        params:    optional dict of metadata.
    """
    save_map(traj_dict, filename, params=params)


def load_trajectory(filename):
    """Load HDF5 trajectory file.

    Returns:
        (traj_dict, params_dict)
    """
    return load_map(filename)


# ============================================================
# YAML parameter files
# ============================================================

def load_params(filename):
    """Load a YAML (or JSON) parameter file.

    Args:
        filename: str -- path to .yaml or .json file.

    Returns:
        dict
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for load_params. Install with: pip install pyyaml"
        )
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def save_params(params, filename):
    """Save a parameter dict to YAML.

    Args:
        params:   dict -- parameters to write.
        filename: str  -- output path.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for save_params. Install with: pip install pyyaml"
        )
    with open(filename, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, allow_unicode=True)


# ============================================================
# Sinusoidal substrate parameter helper
# ============================================================

# Coefficients from Vanossi, Manini, Tosatti, PNAS 109, 16429 (2012).
_SIN_PRESETS = {
    2: (1.0,             0.0),          # lines
    3: (4.0 / 3.0,      0.0),          # triangular
    4: (2.0 ** 0.5,     np.pi / 4.0),  # square
    5: (2.0,            0.0),          # quasicrystal 5-fold
    6: (4.0 / 3.0**0.5, -np.pi / 6.0), # quasicrystal 6-fold
}


def make_sin_params(n_symmetry, spacing):
    """Build substrate parameter dict for an n-fold sinusoidal potential.

    Equivalent to running create_PW_sub.py from the old snips/ directory.

    Args:
        n_symmetry: int   -- fold symmetry (2, 3, 4, 5, or 6).
        spacing:    float -- lattice spacing (same units as cluster positions).

    Returns:
        dict with keys 'well_shape' and 'ks' ready to merge into a full
        params dict.

    Raises:
        ValueError: if n_symmetry is not in {2, 3, 4, 5, 6}.
    """
    if n_symmetry not in _SIN_PRESETS:
        raise ValueError(
            "n_symmetry must be one of %s; got %d."
            % (sorted(_SIN_PRESETS), n_symmetry)
        )
    from flake.substrate import get_ks
    c_n, alpha_n = _SIN_PRESETS[n_symmetry]
    ks = get_ks(float(spacing), n_symmetry, c_n, alpha_n)
    return {'well_shape': 'sin', 'ks': ks.tolist()}


# ============================================================
# Internal HDF5 attribute helpers
# ============================================================

def _write_attrs(h5obj, params):
    """Write params dict as HDF5 attributes on h5obj.

    Scalars and strings are stored directly.  Everything else (lists,
    dicts, numpy arrays) is JSON-encoded as a string attribute.
    """
    for key, val in params.items():
        if isinstance(val, (int, float, str, bool)):
            h5obj.attrs[key] = val
        else:
            # JSON handles lists, dicts, and nested structures.
            # numpy arrays are converted to lists first.
            h5obj.attrs[key] = json.dumps(
                val if not isinstance(val, np.ndarray) else val.tolist()
            )


def _read_attrs(h5obj):
    """Read HDF5 attributes back into a Python dict.

    Attempts JSON decoding for string attributes that look like JSON.
    """
    params = {}
    for key, val in h5obj.attrs.items():
        if isinstance(val, (bytes, str)):
            sval = val.decode('utf-8') if isinstance(val, bytes) else val
            try:
                params[key] = json.loads(sval)
            except (json.JSONDecodeError, ValueError):
                params[key] = sval
        else:
            params[key] = val
    return params
