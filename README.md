
![logo](https://raw.githubusercontent.com/LamaKing/flake_rigid/main/docs/source/_static/logo_html.png)

# FLAKE — Friction and Lattice Analysis of Kinetics and Energetics

FLAKE computes the interlocking potential between a periodic substrate and a finite-size adsorbate, in the rigid approximation.
The adsorbate is treated as a rigid body at a given orientation $\theta$ and center of mass (CM) position $x_\mathrm{cm}, y_\mathrm{cm}$.

This package implements the physical system and automates routines needed to study the statics, dynamics, and scaling laws of friction at nanoscale interfaces.
It was developed at SISSA, Trieste, Italy in the group of Prof. Erio Tosatti and Andrea Vanossi, based on experiments by Xin Cao and Clemens Bechinger at the University of Konstanz, Germany.

See the [documentation](https://flake-rigid.readthedocs.io/en/latest/) or the [source on GitHub](https://github.com/LamaKing/flake_rigid) for more details.

## Installation

**With pip** (recommended):

```console
pip install flake-rigid
```

**With conda** (creates a dedicated environment):

```console
conda create -n flake python=3.11
conda activate flake
pip install flake-rigid
```

**From source** (for development):

```console
git clone https://github.com/LamaKing/flake_rigid.git
cd flake_rigid
pip install -e ".[dev]"
```

Numba will JIT-compile the hot loops on first run; subsequent runs are fast.

### Jupyter notebook kernel

To use FLAKE inside Jupyter notebooks from a conda environment:

```console
conda activate flake
pip install ipykernel
python -m ipykernel install --user --name flake --display-name "Python (flake)"
```

Then select the **Python (flake)** kernel when opening the notebooks in `examples/`.

## Quick start

```console
# Compute a translational energy map
flake map -i my_params.yaml --grid grid_trasl.yaml -o map_trasl.h5

# Find the minimum energy path between two configurations
flake string -i my_params.yaml --cfg string_roto.yaml -o mep.h5

# Run a sweep of MD trajectories over a force grid
flake sweep -i my_params.yaml --spec sweep_Fx.yaml
```

See `example_cli/` for working YAML examples of all three subcommands.

## Substrate

The substrate is defined as a periodic function, either a superposition of plane waves or a potential well repeated on a lattice.
The relevant module is `flake.substrate`.

For a plane-wave (sinusoidal) substrate, the substrate is defined by a set of wave vectors: the number of vectors controls the symmetry and the length sets the spacing [1].
For a lattice of wells, the substrate is defined by the well shape parameters and the lattice vectors [2–5].
These substrate can be decorated with a multi-site basis.

See `examples/0-Substrate_types.ipynb` for details.

## Cluster

The cluster is a collection of points (optionally decorated with a basis) belonging to a 2D Bravais lattice, cut to a given shape (circle, hexagon, rectangle, triangle, ellipse, or arbitrary polygon via Shapely).
The relevant module is `flake.cluster`.

See `examples/1-Cluster_creation.ipynb` for details.

## Static maps

The module `flake.maps` provides translational, rotational, and roto-translational energy landscapes:

- **Translational**: $E(x_\mathrm{cm}, y_\mathrm{cm})$ at fixed $\theta$
- **Rotational**: $E(\theta)$ at fixed CM
- **Roto-translational**: global minimum search in $(x_\mathrm{cm}, y_\mathrm{cm}, \theta)$ space

See `examples/2-Cluster_on_substrate.ipynb` for details.

## Barrier finding

The minimum energy path (MEP) between two configurations in $(x_\mathrm{cm}, y_\mathrm{cm})$ or $(x_\mathrm{cm}, y_\mathrm{cm}, \theta)$ space is found with the string method [6] implemented in `flake.string_method`.
The maximum energy along the MEP gives the static friction force $F_s = \max_s |\nabla E|$.

See `examples/3-Barrier_from_string.ipynb` for details.

## Molecular dynamics

`flake.dynamics.run_md` integrates the overdamped Langevin equation:

$$\eta_t \frac{d\mathbf{r}_\mathrm{cm}}{dt} = \mathbf{F}_\mathrm{ext} + \mathbf{F}_\mathrm{sub} + \boldsymbol{\xi}_t$$

$$\eta_r \frac{d\theta}{dt} = \tau_\mathrm{ext} + \tau_\mathrm{sub} + \xi_r$$

The drag coefficients $\eta_t = N\eta$ and $\eta_r = \eta \sum_i r_i^2$ are computed from the cluster geometry by `calc_cluster_langevin`.
Noise satisfies the fluctuation-dissipation theorem: $\langle \xi_i \xi_j \rangle = 2k_BT\eta_i\,\delta_{ij}\delta(t-t')$.
Sweep infrastructure (`flake.sweep`) parallelises trajectories over a grid of external drives.

See `examples/4-Dynamics.ipynb` for depinning sweeps.

## Scaling laws

`examples/5-Scaling_laws.ipynb` demonstrates:

- Commensurate contact: $F_s \propto N$ (all particles contribute coherently)
- Incommensurate contact: $F_s \propto \sqrt{N}$ (structural superlubricity)

## Units

No internal units conversion is performed; the user chooses a coherent set.

**Colloidal experiments** [2, 3]:
- energy: zJ = $10^{-21}$ J
- length: $\mu$ m
- mass: fg = $10^{-15}$ g
- force: fN, torque: fN·$\mu$m, time: ms

**Nanoscale / AFM experiments**:
- energy: eV
- length: Å
- force: eV/Å $\approx$ 1.6 nN

## References

1. Vanossi, Manini, Tosatti. *Proc. Natl. Acad. Sci.* **109**, 16429 (2012). https://doi.org/10.1073/pnas.1213930109
2. Panizon, Silva et al. *Nanoscale* **15**, 1299 (2023). https://doi.org/10.1039/D2NR04532J
3. Cao, Silva et al. *Phys. Rev. X* **12**, 021059 (2022). https://doi.org/10.1103/PhysRevX.12.021059
4. Cao et al. *Nature Physics* **15**, 776 (2019). https://doi.org/10.1038/s41567-019-0515-7
5. Cao et al. *Phys. Rev. E* **103**, 012606 (2021). https://doi.org/10.1103/PhysRevE.103.012606
6. E, Ren, Vanden-Eijnden. *J. Chem. Phys.* **126**, 164103 (2007). https://doi.org/10.1063/1.2720838
