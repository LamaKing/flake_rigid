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

With conda
----------

If you use conda (or mamba), the recommended approach is to create a dedicated
environment so that Numba and its dependencies do not conflict with other projects:

.. code-block:: console

   conda create -n flake python=3.11
   conda activate flake
   pip install flake-rigid

Verify the installation:

.. code-block:: console

   flake --help

From source
-----------

Source code is hosted at `https://github.com/LamaKing/flake_rigid <https://github.com/LamaKing/flake_rigid>`_.

Clone the repository and install in editable mode (recommended for development):

.. code-block:: console

   git clone https://github.com/LamaKing/flake_rigid.git
   cd flake_rigid
   pip install -e ".[dev]"

This registers the ``flake`` command-line entry point and installs pytest for
running the test suite:

.. code-block:: console

   python -m pytest tests/ -q

Numba will JIT-compile the hot loops on the first run; subsequent runs are fast.

Running the example notebooks
------------------------------

The ``examples/`` folder contains Jupyter notebooks covering all major features.
If you installed FLAKE in a conda environment, you need to register it as a
Jupyter kernel before the notebooks can use it:

.. code-block:: console

   conda activate flake
   pip install ipykernel
   python -m ipykernel install --user --name flake --display-name "Python (flake)"

Then launch Jupyter and select the **Python (flake)** kernel:

.. code-block:: console

   jupyter notebook examples/

If you installed FLAKE directly into your base Python (via ``pip install flake-rigid``
without a conda environment), no kernel registration is needed — the notebooks
will use the default kernel.
