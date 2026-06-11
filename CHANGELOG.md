# Changelog

All notable changes to FLAKE are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] -- 2026-06-12

Complete rewrite from the research prototype (0.0.x).  API breaks with
the old version are intentional; there are no backward-compatibility shims.

### Package

- Renamed source tree to `flake/` (installable via `pip install -e .`).
- Module renames: `tool_create_substrate` -> `substrate`, `tool_create_cluster`
  -> `cluster`, `sweep_md` -> `sweep`, `misc` -> `plot`, `slides_io` -> `io`.
- CLI entry point: `drift` (calls `flake.cli:main`).

### Substrate (`flake.substrate`)

- Full rewrite with Numba `@njit` JIT cores for sin, Gaussian, and tanh
  well shapes; torque accumulation bug fixed in the process.
- `substrate_from_params` factory: returns `(pen_func, en_func, en_inputs=[])`
  closures; all parameters pre-converted to float64 at construction time.
- `_jit_core` and `_jit_params` attributes attached to `en_func` for the
  full-JIT MD loop path (no Python/JIT boundary per step).
- Added `well_shape='flat'` substrate for analytic testing (zero energy,
  zero force, zero torque everywhere).
- `get_ks`: wave-vector generator; must always be used to build `ks` for
  sin substrates (hand-written `ks` produce wrong landscapes silently).

### Cluster (`flake.cluster`)

- Full rewrite: `make_cluster` replaces the old `create_cluster`; supports
  circle, hexagon, rectangle, triangle, parallelogram, ellipse shapes.
- `cluster_from_params` for YAML-driven construction.
- `calc_cluster_langevin`: returns `(eta_t, eta_r)` from single-particle
  drag coefficient and cluster geometry.
- Angles at all public interfaces in degrees; radians only in JIT internals.
- `rotate` applies rotation about an arbitrary centre (default origin).

### Dynamics (`flake.dynamics`)

- Euler-Maruyama overdamped Langevin integrator (`run_md`).
- Full-JIT inner loop (`_md_loop_njit`): when no `stop_fn`/`output_fn`
  callbacks are present, the entire EM loop runs in Numba with no
  Python overhead per step (~2x speedup over Python loop at large N).
- Python-loop path retained for callback support (`stop_fn`, `output_fn`).
- Fluctuation-dissipation theorem: noise amplitude
  `B = sqrt(2*kBT / (eta_i * dt))`, variance `2*D_i*dt`. Correct FDT.
- `theta0` / `pos_cm0` as explicit initial-condition kwargs; theta not
  baked into cluster positions.
- Guards: `ValueError` for `eta_r=0` with `Tau!=0` or `kBT>0`;
  `UserWarning` for `kBT=0` (deterministic saddle-point sensitivity).

### Sweep (`flake.sweep`)

- `sweep_md`: parallel parameter sweeps using joblib loky backend.
- `grid_sweep`, `line_sweep`, `force_sweep` spec builders.
- `save=True` writes `run_NNNN-param_val-.../params.yaml` and `traj.h5`;
  directory names include only the parameters that vary across runs.
- Resume logic: skips runs whose `traj.h5` exists unless `overwrite=True`.
- `load_sweep` / `filter_sweep` for reloading saved sweeps.
- Post-processing functions: `drift_velocity`, `mean_velocity`,
  `last_state`, `drift_omega`.

### Maps (`flake.maps`)

- `translational_map`, `rotational_map`, `rototrasl_map`.
- `pos_cm_grid` parameter for sin and quasicrystal substrates (no unit cell).

### String method (`flake.string_method`)

- n-dimensional minimum-energy path finder; coordinate scaling support.
- `StringPath`, `StringPotential`, `find_mep`.

### CLI (`flake.cli`)

- Unified `drift` command: subcommands `map`, `sweep`, `string`,
  `make-params`, `make-sweep`.
- All subcommands read physics from a single `params.yaml`.
- `drift sweep` writes informative run directories and supports `--overwrite`.

### Tests

- 122 unit and integration tests (pytest, `not slow` in ~34 s).
- Physics validation (`tests/test_physics.py`): F1s analytic agreement,
  commensurate linear scaling Fc = N * F1s, superlubricity at 30° rotation.
- JIT benchmark: full-JIT loop faster than Python loop (speedup > 1).
- Parallel sweep benchmark: loky sweep faster than explicit serial loop.

---

## [0.0.x] -- 2024 (research prototype)

Initial research code: translational and rotational static energy maps,
string method for minimum energy paths, early overdamped Langevin MD.
No test suite, no packaging.  Superseded entirely by 0.1.0.
