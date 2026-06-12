Installation
=============

FLAKE is written in Python 3 (>=3.10). Required dependencies (installed automatically):

- numpy
- scipy
- numba

Optional dependencies (for examples and documentation):

- matplotlib
- shapely
- ase (for POSCAR import)
- h5py (for HDF5 map/trajectory files)
- joblib (for parallel sweeps)
- pyyaml (for CLI YAML input)

From source (recommended)
--------------------------

Clone the repository and install in editable mode:

.. code-block:: console

   git clone https://github.com/LamaKing/slides_rigid.git
   cd slides_rigid
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
