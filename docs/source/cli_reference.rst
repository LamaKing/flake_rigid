Command-line interface
======================

The ``flake`` command is the high-level entry point for FLAKE. All five
subcommands are driven by YAML input files, require no Python scripting, and
write results to HDF5 files that can be loaded with :mod:`flake.io`.
For programmatic use and custom analysis see the API modules directly.

.. code-block:: console

   flake <subcommand> [options]
   flake --help
   flake <subcommand> --help

Typical workflow
----------------

A typical session proceeds in this order:

1. **Generate a parameter file** with ``flake make-params``. This writes a
   ``params.yaml`` with correct wave vectors for the substrate (hand-writing
   them is error-prone) and a default cluster block to edit.

2. **Explore the energy landscape** with ``flake map``. Run a translational or
   rotational map to identify the corrugation scale and pick sensible starting
   points and force ranges for the next steps.

3. **Find energy barriers** with ``flake string`` (optional). The string method
   gives the minimum-energy path and static friction force between two
   configurations — useful for understanding what drives depinning before
   committing to a full dynamics sweep.

4. **Run a parameter sweep** with ``flake sweep``. Sweep external force,
   temperature, or any other ``run_md`` argument over a grid of values.
   Results are written to per-run ``run_NNNN-*/`` directories for easy
   resumption and post-processing with ``flake.sweep.load_sweep``.

.. code-block:: console

   # Step 1 — generate params.yaml for a 6-fold sinusoidal substrate
   flake make-params --substrate sin --n 6 --spacing 1.0 -o params.yaml

   # Step 2 — translational energy map over one unit cell
   flake map -i params.yaml --grid grid.yaml -o map.h5

   # Step 3 — minimum energy path between two configurations
   flake string -i params.yaml --cfg string.yaml -o mep.h5

   # Step 4 — depinning sweep over a range of applied forces
   flake sweep -i params.yaml --spec sweep_Fx.yaml --outdir sweep_out/

Working YAML examples for all subcommands are in ``test_cli_and_phys/``.


Subcommand reference
--------------------

``flake map``
~~~~~~~~~~~~~

Compute a static energy map and write the result to HDF5.

.. code-block:: console

   flake map -i PARAMS -o OUTPUT [--grid GRID_YAML]

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i / --input``
     - Physics parameters YAML (**required**).
   * - ``--grid``
     - Grid config YAML. If omitted, sensible defaults are used.
   * - ``-o / --output``
     - Output HDF5 file (default: ``map.h5``).

The map type (translational, rotational, roto-translational) and grid
parameters are set in ``grid.yaml`` — see :ref:`grid-yaml` below.
Output can be loaded with :func:`flake.io.load_map`.

``flake sweep``
~~~~~~~~~~~~~~~

Run one overdamped Langevin MD trajectory per point in a parameter grid and
collect observables.

.. code-block:: console

   flake sweep -i PARAMS --spec SPEC_YAML [--outdir DIR]

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i / --input``
     - Physics parameters YAML (**required**).
   * - ``--spec``
     - Sweep specification YAML (**required**). See :ref:`sweep-yaml`.
   * - ``--outdir``
     - Output directory (overrides ``outdir`` key in the spec YAML).

Each run writes ``run_NNNN-key_val-.../traj.h5`` and ``params.yaml``
under the output directory. Incomplete sweeps can be resumed: runs with an
existing ``traj.h5`` are skipped unless ``overwrite: true`` is set in the
spec. Results are loaded with :func:`flake.sweep.load_sweep`.

``flake string``
~~~~~~~~~~~~~~~~

Find the minimum energy path (MEP) between two configurations using the
string method, and write the path to HDF5.

.. code-block:: console

   flake string -i PARAMS --cfg STRING_YAML -o OUTPUT

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Description
   * - ``-i / --input``
     - Physics parameters YAML (**required**).
   * - ``--cfg``
     - String method configuration YAML (**required**). See :ref:`string-yaml`.
   * - ``-o / --output``
     - Output HDF5 file (default: ``mep.h5``).

The search dimension (2D translational or 3D roto-translational) is inferred
from the length of ``p0`` in the config. Output contains the converged path,
energy profile, and the barrier height
:math:`\Delta E = \max(E) - \min(E)`.

``flake make-params``
~~~~~~~~~~~~~~~~~~~~~

Generate a starter ``params.yaml`` for a given substrate type without running
any physics. Use this to get correct wave vectors for sinusoidal substrates
— hand-writing them produces wrong lattice symmetry with no error.

.. code-block:: console

   flake make-params --substrate sin --n 6 --spacing 1.0 -o params.yaml
   flake make-params --substrate gaussian --spacing 1.0 -o params.yaml
   flake make-params --substrate tanh --spacing 1.0 -o params.yaml

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Description
   * - ``--substrate``
     - Substrate type: ``sin``, ``gaussian``, or ``tanh`` (**required**).
   * - ``--n``
     - Fold symmetry for ``sin`` substrate: 2, 3, 4, 5, or 6 (**required for sin**).
   * - ``--spacing``
     - Substrate lattice spacing in your length units (**required**).
   * - ``-o / --output``
     - Output file (default: ``params.yaml``; use ``-`` for stdout).

``flake make-sweep``
~~~~~~~~~~~~~~~~~~~~

Generate a ``sweep_spec.yaml`` without running any physics. Useful for
building sweep grids from the command line rather than writing YAML by hand.

.. code-block:: console

   flake make-sweep --type grid --grid Fx=0,50,100,150,200 \
       --base eta=1.0 --base kBT=1e-8 --base n_steps=200000 \
       --post-fn drift_velocity -o sweep_Fx.yaml

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``--type``
     - Layout: ``grid`` (Cartesian product), ``line`` (paired rows), or
       ``force`` (fixed-angle force magnitude sweep). Default: ``grid``.
   * - ``--grid KEY=V1,V2,...``
     - Grid axis. Repeatable. Values must be floats.
   * - ``--line KEY=V1,V2,...``
     - Line column. Repeatable. All columns must have equal length.
   * - ``--force-vals V1,V2,...``
     - Comma-separated force magnitudes (for ``--type force``).
   * - ``--phi-deg``
     - Force direction in degrees from x-axis (default: 0).
   * - ``--base KEY=VAL``
     - Base MD kwarg added to ``base_md_kwargs``. Repeatable.
   * - ``--post-fn``
     - Observable extractor: ``drift_velocity``, ``drift_omega``,
       ``mean_velocity``, or ``last_state``.
   * - ``--n-jobs``
     - Number of parallel workers to embed in the spec.
   * - ``--no-save-traj``
     - Set ``save_traj: false`` (omit raw trajectories, keep post_fn results only).
   * - ``--outdir``
     - Output directory to embed in the spec.
   * - ``-o / --output``
     - Output YAML file (default: ``sweep_spec.yaml``).


.. _yaml-files:

YAML file reference
-------------------

All input files use `YAML <https://yaml.org>`_ syntax. This section documents
the allowed keys for each file type.

.. _params-yaml:

Physics input (``params.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single file describing the substrate and the cluster. Passed to every
subcommand via ``-i params.yaml``. Generate a template with
``flake make-params``.

.. code-block:: yaml

   # --- substrate ---
   well_shape: sin          # 'sin', 'gaussian', or 'tanh'
   epsilon: 1.0             # corrugation amplitude (your energy units)

   # sinusoidal substrate: wave vectors from get_ks (required)
   ks:
     - [ 4.1887902,  0.0      ]
     - [-2.0943951,  3.6275987]
     - [-2.0943951, -3.6275987]

   # gaussian/tanh substrate: lattice vectors (required instead of ks)
   # b1: [1.0, 0.0]
   # b2: [0.5, 0.8660254]
   # sigma: 0.3             # well width (gaussian only)
   # wd: 0.1                # wall steepness (tanh only; smaller = sharper)

   sub_basis:               # substrate basis (list of [x, y] offsets)
     - [0.0, 0.0]

   # --- cluster ---
   cluster_shape: circle    # 'circle', 'hexagon', 'triangle',
                            # 'rectangle', 'ellipse', 'parallelogram'
   N1: 9                    # supercell repeat along a1
   N2: 9                    # supercell repeat along a2 (use N1==N2 for hexagon)
   a1: [1.0, 0.0]           # cluster lattice vector 1
   a2: [0.5, 0.8660254]     # cluster lattice vector 2
   cl_basis:                # cluster basis (list of [x, y] offsets)
     - [0.0, 0.0]

   theta: 0.0               # initial orientation in degrees (reference frame)
   pos_cm: [0.0, 0.0]       # ignored at CLI level; set in sweep/string configs

.. note::

   Wave vectors ``ks`` for sinusoidal substrates **should** be generated with
   ``flake make-params --substrate sin``. Hand-written values are prone to error and could result in wrong
   lattice symmetry without raising an error.
   ``theta`` sets the initial cluster orientation for ``flake sweep`` and the
   fixed orientation for a 2D ``flake string`` search; it is **not** applied
   by ``cluster_from_params`` — each command applies it appropriately.

.. _grid-yaml:

Map grid (``grid.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~

Controls the grid for ``flake map``. All keys are optional (defaults shown).

.. code-block:: yaml

   map_type: translational  # 'translational', 'rotational', or 'rototrasl'

   # translational / rototrasl: grid extent
   # For gaussian/tanh substrates use fractional coords (0..1):
   frac_x: [0.0, 1.0]
   frac_y: [0.0, 1.0]
   n_x: 50
   n_y: 50

   # For sinusoidal substrates use Cartesian coords instead:
   # x_range: [-5.0, 5.0]
   # y_range: [-5.0, 5.0]

   # rotational / rototrasl: angular scan
   theta_range: [0.0, 360.0]   # degrees
   n_theta: 36

   n_jobs: 1                   # parallel workers (-1 = all cores)

.. note::

   Sinusoidal and quasicrystal substrates have no unit cell, so fractional
   coordinates are undefined. Use ``x_range`` / ``y_range`` (Cartesian) instead.
   If omitted for ``well_shape: sin``, a default of ±2 substrate periods is
   estimated from ``|k[0]|`` and a warning is emitted.

.. _sweep-yaml:

Sweep specification (``sweep.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls the parameter grid for ``flake sweep``. Generate a template with
``flake make-sweep``.

.. code-block:: yaml

   sweep_type: grid         # 'grid' or 'line'

   # grid sweep: Cartesian product of all listed values
   grid:
     Fx: [0.0, 50.0, 100.0, 150.0, 200.0]
     # Any run_md kwarg can appear here: Fy, Tau, kBT, eta, ...

   # line sweep: paired values (all lists must have the same length)
   # line:
   #   Fx: [0.0, 50.0, 100.0]
   #   Fy: [0.0, 25.0, 50.0]

   base_md_kwargs:          # run_md kwargs applied to every run
     eta: 1.0
     kBT: 1.0e-8
     dt:  1.0e-3
     n_steps: 200000
     print_every: 500

   post_fn: drift_velocity  # 'drift_velocity', 'drift_omega',
                            # 'mean_velocity', or 'last_state'
   save_traj: true          # save full trajectory to traj.h5
   overwrite: false         # skip runs with existing traj.h5 (resume)
   n_jobs: -1               # parallel workers (-1 = all cores)
   backend: loky            # joblib backend
   outdir: sweep_out        # output directory (overrides CLI --outdir)

The valid keys for ``grid`` / ``line`` entries and ``base_md_kwargs`` are the
keyword arguments of :func:`flake.dynamics.run_md`:
``eta``, ``Fx``, ``Fy``, ``Tau``, ``kBT``, ``dt``, ``n_steps``,
``theta0``, ``pos_cm0``, ``print_every``, ``seed``.

.. _string-yaml:

String method configuration (``string.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controls ``flake string`` (minimum-energy path search).

.. code-block:: yaml

   # endpoints in (x, y) for 2D, or (x, y, theta_deg) for 3D
   p0: [0.0, 0.0,  0.0]
   p1: [1.0, 0.0, 60.0]

   # arc-length metric scales: [lx, ly] or [lx, ly, l_theta]
   # l_theta sets how many degrees equals one length unit.
   # Rule of thumb: l_theta ~ lattice_spacing * 60 / (2*pi)
   scale: [1.0, 1.0, 60.0]

   n_points: 200            # number of images along the path
   n_iter:   5000           # maximum string iterations
   step:     1.0e-4         # gradient-descent step size
   tol:      1.0e-3         # convergence tolerance (max pointwise displacement,
                            # in raw unscaled coordinates)
   fix_ends: true           # keep p0 and p1 fixed during relaxation

.. note::

   ``theta`` values in ``p0`` and ``p1`` are in **degrees**.
   The ``scale`` parameter affects arc-length reparametrization only, not the
   gradient step size. In 3D mode, ``tol`` is measured in raw coordinates
   (degrees for the theta component), so set it relative to the translational
   scale rather than the arc-length metric.
   For a triangular-lattice contact a good starting point is
   ``scale: [a, a, 60]`` where ``a`` is the substrate lattice spacing.
