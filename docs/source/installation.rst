Installation
=============

FLAKE is written in Python 3 (>=3.10). Required dependencies (installed automatically):

- numpy
- scipy
- numba
- pyyaml (CLI YAML input)
- h5py (HDF5 map/trajectory files)
- joblib (parallel sweeps)

Optional dependencies (for examples and documentation):

- matplotlib
- shapely
- ase (for POSCAR import)

From PyPI
---------

.. code-block:: console

   pip install flake-rigid

This installs the ``flake`` command-line entry point and all required dependencies.

From source
-----------

Source code is hosted at `https://github.com/LamaKing/flake_rigid <https://github.com/LamaKing/flake_rigid>`_.

Clone the repository and install in editable mode (recommended for development):

.. code-block:: console

   git clone https://github.com/LamaKing/flake_rigid.git
   cd flake_rigid
   pip install -e ".[dev]"

This registers the ``flake`` command-line entry point and installs pytest for running the test suite.

Verify the installation:

.. code-block:: console

   flake --help
   python -m pytest tests/ -q

Numba will JIT-compile the hot loops on the first run; subsequent runs are fast.

Examples
--------

The ``examples/`` folder contains Jupyter notebooks covering all major features.
Open them with:

.. code-block:: console

   jupyter notebook examples/
