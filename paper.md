---
title: 'FLAKE: Friction and Lattice Analysis of Kinetics and Energetics of rigid 2D clusters on crystalline substrates'
tags:
  - Python
  - friction
  - superlubricity
  - nanotribology
  - Langevin dynamics
authors:
  - name: Andrea Silva
    orcid: 0000-0001-6699-8115
    affiliation: "1, 2"
affiliations:
  - index: 1
    name: SISSA – Scuola Internazionale Superiore di Studi Avanzati, Trieste, Italy
    ror: 02k6dtp35
  - index: 2
    name: CNR-IOM, Trieste, Italy
    ror: 01460ce75
date: 13 June 2026
bibliography: paper.bib
---

# Summary

When a small solid island slides across a crystalline surface, the outcome —
smooth flow or stick-slip jerking, free rotation or angular locking — is
controlled almost entirely by the geometric relationship between the island
and the surface: their relative orientation, size, and lattice symmetry.
These phenomena, collectively studied under the banner of *structural
superlubricity* [@hirano_superlubricity_1990; @dienwiebel_superlubricity_2004],
are central to nanoscale friction and to the interpretation of colloidal
monolayer experiments [@cao_orientational_2019; @cao_pervasive_2021].

FLAKE (Friction and Lattice Analysis of Kinetics and Energetics) is a
Python package for simulating the statics and overdamped Langevin dynamics
of a rigid 2D cluster of particles sliding over a periodic or quasiperiodic
substrate potential. It provides: energy landscape maps over translational
and rotational degrees of freedom; minimum-energy path (MEP) search via the
string method in both 2D (translational) and 3D (translational + rotational)
configuration space; and parameter-sweep molecular dynamics for depinning
thresholds and velocity–force scaling laws. All substrate types — sinusoidal,
Gaussian well, and tanh well — as well as cluster geometries (circle,
hexagon, triangle, rectangle, ellipse) are fully configurable through a
YAML parameter file and a command-line interface.

# Statement of Need

Nanotribology experiments with colloidal clusters
[@cao_orientational_2019; @cao_pervasive_2021] and van der Waals
heterostructures [@cao_moire_2022; @liao_twisting_2023] produce rich
phenomenology — orientational locking, rotational depinning, coupled
roto-translational motion — that is difficult to interpret without a
corresponding simulation tool. The standard computational approach in this
field is all-atom molecular dynamics with realistic interatomic potentials,
typically using LAMMPS [@thompson_lammps_2022]. While indispensable for
elastic effects and chemical specificity, all-atom simulations are
prohibitively expensive for the systematic parameter sweeps (cluster size,
orientation, substrate symmetry, applied force) needed to extract scaling
laws or build phase diagrams. They also entangle rigidity-breaking elastic
deformation with the purely geometric commensurability effects that dominate
in stiff 2D materials.

FLAKE occupies the complementary rigid-body niche. Treating the cluster as
perfectly rigid reduces the many-body problem to three collective degrees of
freedom — two translational and one rotational — and makes it possible to
run thousands of trajectories in seconds on a laptop. This is the regime
relevant to stiff flakes (graphene, hBN, MoS$_2$) on flat substrates at
moderate loads, and to colloidal clusters in optical traps or on patterned
surfaces, where interparticle distances are effectively fixed. FLAKE is
designed to be a fast, Python-native foundation for experiment interpretation
and as a rigid-body baseline for LAMMPS elastic-contact studies, not a
replacement for them.

General-purpose MD packages such as ASE [@larsen_ase_2017] or OpenMM
support overdamped integrators but are designed around per-atom force fields
and do not expose the collective rigid-body geometry needed here. Implementing
roto-translational Langevin dynamics, substrate closure factories, and MEP
search for orientation-dependent energy barriers on top of those frameworks
would require more custom code than the package itself.

The need for a dedicated rigid-body tribology tool has become increasingly
acute. Recent years have seen a wave of independent computational studies
addressing exactly the phenomena FLAKE targets — rotational energy landscapes
and scaling laws for twisted bilayers
[@zhang_universal_2025; @zhang_frictionless_prl_2025; @zhang_directional_2026],
shape-dependent friction scaling [@yan_shape_2024; @gao_frictional_2025],
orientational control via strain [@zhou_orientational_2025], and experimental
evidence for directional superlubricity and anomalous diffusion in van der
Waals heterostructures [@lester_directional_2024]. These studies use a
variety of tools and conventions, often relying on LAMMPS with
material-specific force fields that constrain the accessible geometries and
substrate symmetries. Reviews of the scaling-up challenge
[@ying_scalingup_2025] highlight that systematic parameter exploration
remains a bottleneck. FLAKE aims to provide the missing common ground: a
single, geometry-agnostic, Python-native package that makes such systematic
studies straightforward.

# State of the Field

Structural superlubricity in 2D materials is an active research front
[@muser_structural_2023; @ying_scalingup_2025]. The field has reached a point
where the geometric, rigid-body mechanism is well-established in principle,
but quantitative predictions — scaling of the static friction force with
cluster size and shape, the role of orientation, the nature of
roto-translational coupling — still require systematic numerical exploration
that is expensive with fully elastic, force-field-based MD. FLAKE goes beyond
existing approaches by:

- supporting arbitrary cluster shapes and sizes with numerically exact energy
  and force evaluation via Numba-JIT substrate potentials [@lam_numba_2015];
- coupling translational and rotational degrees of freedom in a single
  Euler–Maruyama integrator with fluctuation-dissipation-theorem-correct
  noise [@risken_fokker_planck_1989];
- providing a roto-translational string method [@weinan_string_2002] to find
  minimum-energy paths in the joint (x, y, $\theta$) space — an operation
  that has no equivalent in existing open software to the author's knowledge;
- wrapping everything in a sweep infrastructure (via joblib [@joblib_2024])
  that makes scaling-law studies a one-command operation.

**DFT-level PES tools** such as TribChem [@losi_tribchem_2023] and the
high-throughput workflow of @wolloch_highthroughput_2022 operate at the
opposite end of the length-scale spectrum: they compute interfacial adhesion
and shear strength from first-principles electronic structure, also in the
rigid approximation, but for unit-cell-sized systems with full chemical
specificity. These tools are naturally complementary to FLAKE — their output
(the corrugation amplitude $\epsilon$ and the shape of the well) can
parametrise FLAKE's substrate potentials for specific material interfaces.

**Geometric descriptors** such as the Registry Index
[@hod_registry_2013; @cao_registry_tmd_2022] similarly characterise
interfacial commensurability from atomic geometry alone, producing
qualitative energy maps analogous to FLAKE's static maps. However, they
carry no energy scale and cannot be used to drive dynamics: they describe
*which* configurations are commensurate but cannot compute the barrier
between them or the resulting friction force.

In LAMMPS simulations of 2D-material interfaces, the interlayer interaction
(the analogue of FLAKE's substrate potential) is typically described by the
Interlayer Potential (ILP) of @leven_ilp_2016, which uses Gaussian-shaped
atom-atom overlap terms for the registry-dependent repulsion. This form is
qualitatively captured by FLAKE's Gaussian-well substrate — providing a
physical justification for the substrate model when the code is used to
interpret LAMMPS-parametrised systems.

No existing Python package targets this combination of features for the
rigid-body tribology use case.

# Software Design

## Overdamped Langevin dynamics

The equation of motion for the cluster center-of-mass position $\mathbf{r}$
and orientation angle $\theta$ is:

$$\eta_t \dot{\mathbf{r}} = \mathbf{F}_\text{sub}(\mathbf{r}, \theta) + \mathbf{F}_\text{ext} + \boldsymbol{\xi}_t$$

$$\eta_r \dot{\theta} = \tau_\text{sub}(\mathbf{r}, \theta) + \tau_\text{ext} + \xi_r$$

where $\eta_t$ and $\eta_r$ are translational and rotational drag coefficients
computed from cluster geometry, $\mathbf{F}_\text{sub}$ and $\tau_\text{sub}$
are the total substrate force and torque on the rigid cluster, and
$\boldsymbol{\xi}_{t,r}$ are Gaussian white noise terms satisfying the
fluctuation-dissipation theorem:
$\langle \xi_i(t)\xi_j(t') \rangle = 2 k_B T \eta_i \delta_{ij}\delta(t-t')$.

The Euler–Maruyama discretisation with time step $\Delta t$ uses a
noise amplitude $B_i = \sqrt{2k_BT/(\eta_i \Delta t)}$, giving variance
$B_i^2 \Delta t = 2 k_B T / \eta_i$ per step — the correct FDT scaling.
An earlier version of this code used $B_i = \sqrt{2 k_B T \eta_i}$ (a
common but incorrect form), which produced noise variance proportional to
$\Delta t^2$ instead of $\Delta t$; this bug has been corrected in v0.1.0.

## JIT substrate closures and performance

All substrate energy and force kernels are implemented as Numba
just-in-time compiled functions (`@njit`) [@lam_numba_2015]. The `substrate_from_params` factory
pre-converts all parameters to float64 arrays once at construction time and
returns closures, eliminating per-step Python overhead. When no Python
callback is needed (no custom stop or output functions), `run_md` dispatches
to a fully JIT-compiled inner loop `_md_loop_njit`, reducing the per-step
cost to pure native code.

## String method for minimum-energy paths

`find_mep` implements the zero-temperature string method
[@weinan_string_2002] in 2D (translational MEP) and 3D
(roto-translational MEP). The arc-length metric in 3D uses independent
length scales $(l_x, l_y, l_\theta)$ to put translational and angular
displacements on equal footing — critical for finding physically meaningful
paths when rotation and translation are coupled.

## CLI and sweep infrastructure

A YAML-driven command-line interface (`flake map`, `flake sweep`,
`flake string`) makes it possible to run energy maps, MD sweeps, and string
calculations without writing Python. `sweep_md` parallelises over a
parameter grid using joblib's loky backend [@joblib_2024], names output
directories by the varying parameters, and supports resuming interrupted
sweeps.

# Research Impact

FLAKE v0.1.2 is the formalized, tested, and packaged release of a codebase
developed from 2022 onward at SISSA. The predecessor (v0.0.6, publicly
available since November 2022 with ReadTheDocs documentation) was the
primary simulation tool for the following published studies:

- @silva_frictionless_2023 (Nanoscale): translational and rotational
  superlubricity of rigid clusters on sinusoidal substrates; scaling laws
  $F_s \propto N$ (commensurate) and $F_s \propto \sqrt{N}$ (incommensurate).
- @cao_moire_2022 (Physical Review X): coupled roto-translational depinning
  and the role of Moiré pattern evolution in colloidal and van der Waals
  systems.
- @liao_twisting_2023 (ACS Applied Materials and Interfaces): twisting
  dynamics of large lattice-mismatch van der Waals heterostructures.

The colloidal experiments interpreted in @cao_orientational_2019 (Nature
Physics) and @cao_pervasive_2021 (Physical Review E) motivated the
roto-translational coupling design of the package.

Version 0.1.2 adds a correct FDT noise implementation, a full-JIT MD inner
loop (10–50x faster than the v0.0.6 Python loop for long trajectories), the
roto-translational string method, a flat substrate option for testing, and a
complete test suite (125 unit and physics tests). The package is currently
in active use for ongoing research projects on quasicrystalline substrates
and angular depinning at SISSA.

# AI Usage Disclosure

Claude Code (Anthropic, `claude-sonnet-4-6`) was used during the development
of v0.1.0 for: module restructuring and API cleanup (renaming `src/` to
`flake/`, updating all import paths); documentation (module docstrings with
`.. math::` RST blocks, README, Sphinx configuration); test scaffolding
(generating test stubs for new functions); and paper drafting (initial text
of this manuscript). The physical model, numerical algorithms, substrate
potential implementations, string method, and overall software architecture
were designed by the author and were present in the predecessor codebase
prior to AI assistance. All AI-generated outputs were reviewed, edited, and
validated by the author. The core physics correctness — including the FDT
noise fix, the torque-arm convention, and the roto-translational arc-length
metric — was identified, verified, and decided by the author.

# Acknowledgements

The author thanks Emanuele Panizon, Erio Tosatti, Andrea Vanossi,
Nicola Manini, and Clemens Bechinger for the physics discussions and
experimental collaborations that motivated and shaped this work.

This work was supported by the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation programme, grant
agreement No. 834402 (ULTRADISS), and by the Italian Ministry of University
and Research (MUR) under PRIN grant No. 20178PZCB5 (UTFROM).

# References
