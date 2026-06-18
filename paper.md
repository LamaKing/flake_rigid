---
title: 'FLAKE: Friction and Lattice Analysis of Kinetics and Energetics of rigid 2D clusters on crystalline substrates'
tags:
  - python
  - condensed matter physics
  - friction
  - superlubricity
  - nanotribology
  - Langevin dynamics
  - moire pattern
authors:
  - name: Andrea Silva
    orcid: 0000-0001-6699-8115
    affiliation: "1, 2"
affiliations:
  - index: 1
    name: SISSA – Scuola Internazionale Superiore di Studi Avanzati, Trieste, Italy
    ror: 004fze387
  - index: 2
    name: CNR-IOM, Trieste, Italy
    ror: 00yfw2296 
date: 13 June 2026
bibliography: paper.bib
---

# Summary

When a small solid island slides across a crystalline surface, the outcome —
smooth flow or stick-slip jerking, free rotation or angular locking — is
controlled almost entirely by the geometric relationship between the island
and the surface: their relative orientation, size, and lattice symmetry.
These phenomena are central to nanoscale friction and to the interpretation of colloidal
monolayer experiments [@cao_orientational_2019; @cao_pervasive_2021] and are
the premises for the study of *structural superlubricity* in real interfaces
[@wang_colloquium_2024; @hirano_superlubricity_1990; @dienwiebel_superlubricity_2004; @ying_scalingup_2025; @yao_superlubricity_2025].

FLAKE (Friction and Lattice Analysis of Kinetics and Energetics) is a
Python package for simulating the statics and Langevin dynamics of a rigid
2D cluster sliding over a periodic or quasiperiodic substrate potential.
It is designed for researchers in nanotribology and 2D-materials physics who
need fast, systematic exploration of how cluster geometry, orientation, and
substrate symmetry control the energy landscape and friction — studies that
are prohibitively expensive with fully atomistic simulations. FLAKE has been the primary simulation tool
for three published studies on structural superlubricity and colloidal friction
[@silva_frictionless_2023; @cao_moire_2022; @liao_twisting_2023].

# Statement of Need

Nanotribology experiments with colloidal clusters
[@cao_orientational_2019; @cao_pervasive_2021; @cao_moire_2022] and van der Waals
heterostructures [@liao_twisting_2023; @yao_superlubricity_2025] produce rich
phenomenology — orientational locking, rotational depinning, coupled
roto-translational motion — that is difficult to interpret without a
corresponding simulation tool. The standard computational approach in this
field is all-atom molecular dynamics with realistic interatomic potentials,
typically using LAMMPS [@thompson_lammps_2022]. While indispensable for
elastic effects and chemical specificity, all-atom simulations are
prohibitively expensive for the systematic parameter sweeps needed to extract
scaling laws and build phase diagrams. They also entangle elastic deformation
with the purely geometric commensurability effects that dominate in stiff 2D
materials. General-purpose MD packages such as ASE [@larsen_ase_2017] support
overdamped integrators but are built around per-atom force fields and do not
expose the collective rigid-body geometry needed here.

Recent years have seen a wave of independent computational studies addressing
exactly the phenomena FLAKE targets — rotational energy landscapes and scaling
laws for twisted bilayers
[@zhang_universal_2025; @zhang_frictionless_prl_2025; @zhang_directional_2026],
shape-dependent friction scaling [@yan_shape_2024; @gao_frictional_2025],
orientational control via strain [@zhou_orientational_2025], and experimental
evidence for directional superlubricity in van der Waals heterostructures
[@lester_directional_2024]. These studies use a variety of tools and
conventions, often relying on LAMMPS with material-specific force fields that
constrain the accessible geometries and substrate symmetries. Reviews of the
scaling-up challenge [@ying_scalingup_2025] highlight that systematic parameter
exploration remains a bottleneck. FLAKE provides the missing common ground: a
single, geometry- and chemistry-agnostic, Python-native rigid-body package
designed for fast, systematic friction studies, and a clean baseline for
LAMMPS elastic-contact studies.

# State of the Field

Structural superlubricity in 2D materials is an active research front
[@wang_colloquium_2024; @ying_scalingup_2025]. The geometric, rigid-body
mechanism is now well-established in principle [@silva_frictionless_2023],
but quantitative predictions — friction scaling with cluster size and shape,
the role of orientation, roto-translational coupling — still require systematic
numerical exploration. No existing open Python package targets this combination
for the rigid-body tribology use case, leaving each research group to implement
its own bespoke simulation protocol.

**DFT-level PES tools** such as TribChem [@losi_tribchem_2023] and the
high-throughput workflow of @wolloch_highthroughput_2022 operate at the
opposite length-scale extreme: they compute interfacial adhesion and shear
strength from first-principles electronic structure for periodic unit-cell-sized systems.
These tools are naturally complementary to FLAKE — their output (corrugation
amplitude $\epsilon$ and well shape) can parametrise FLAKE's substrate
potentials for specific material interfaces.

**Geometric descriptors** such as the Registry Index
[@hod_registry_2013] characterise interfacial
commensurability from atomic geometry alone, producing qualitative maps.
However, they carry no energy scale and cannot drive dynamics or compute energy
barriers.

In LAMMPS simulations of 2D-material interfaces, the interlayer interaction
is typically described by the Interlayer Potential (ILP) of @leven_ilp_2016,
which uses Gaussian-shaped atom-atom overlap terms for registry-dependent
repulsion — qualitatively captured by FLAKE's Gaussian-well substrate,
providing a physical link between the two approaches.

# Software Design

The central design choice in FLAKE is the rigid-body reduction.
A cluster of $N$ particles on a substrate in principle requires $2N$ or $3N$
equations of motion. Treating the cluster as a rigid body collapses this to
three collective degrees of freedom — center-of-mass position
$\mathbf{r} = (x, y)$ and orientation angle $\theta$ — governed by:

$$\eta_t \dot{\mathbf{r}} = \mathbf{F}_\text{sub}(\mathbf{r}, \theta) + \mathbf{F}_\text{ext} + \boldsymbol{\xi}_t$$

$$\eta_r \dot{\theta} = \tau_\text{sub}(\mathbf{r}, \theta) + \tau_\text{ext} + \xi_r$$

where $\eta_t = N\eta$ and $\eta_r = \eta \sum_i r_i^2$ are translational and rotational drag coefficients
computed from cluster geometry, $\mathbf{F}_\text{sub}$ and $\tau_\text{sub}$
are the total substrate force and torque summed over all particles, and
$\boldsymbol{\xi}_{t,r}$ are Gaussian white noise terms:
$\langle \xi_i(t)\xi_j(t') \rangle = 2 k_B T \eta_i \delta_{ij}\delta(t-t')$.
The overdamped (inertia-free) limit is the physically appropriate regime for
colloidal clusters in viscous media [@cao_orientational_2019] and for 2D
materials at low sliding speeds; extending the integrator to include inertia
is straightforward.
This reduction makes it possible to run thousands of trajectories in minutes
on a laptop, enabling the systematic parameter sweeps — over cluster size,
shape, orientation, substrate symmetry, and applied force — that are the
primary target of FLAKE.

The rigidity assumption is justified by the physics: for stiff 2D materials
(graphene, hBN, MoS$_2$) at moderate loads, and for colloidal clusters where
interparticle distances are effectively fixed, the dominant friction mechanism
is geometric commensurability, not elastic deformation [@silva_frictionless_2023].
**Limitations.** The rigid-body approximation does not capture intralayer
elasticity, which drives the Aubry-type commensurate–incommensurate
transition [@wang2025aubry]; moiré-induced out-of-plane buckling in
freestanding bilayers [@wang_buckling_2024]; or elastic edge effects that
give rise to chiral moiré dynamics at finite contacts [@silva_chirality_2024].
These effects require fully elastic simulations and are explicitly beyond
FLAKE's scope.

The performance design follows directly from this reduction.
All substrate energy and force kernels are Numba just-in-time compiled
(`@njit`) [@lam_numba_2015]: the `substrate_from_params` factory pre-converts
parameters to float64 arrays once and returns closures, eliminating per-step
Python overhead, and `run_md` dispatches to a fully JIT-compiled inner loop
when no Python callbacks are needed (10–50$\times$ faster than an interpreted
loop for long trajectories).


The string method for minimum-energy paths [@weinan_string_2002] is extended
to the joint $(x, y, \theta)$ space using independent length scales
$(l_x, l_y, l_\theta)$ that put translational and angular displacements on
equal footing — a non-trivial metric choice required to find physically
meaningful roto-translational paths.

**Features.** FLAKE provides:

- *Energy landscape maps* over translational and/or rotational degrees of
  freedom (`flake map`), for static analysis of commensurability and
  orientational locking.
- *Minimum-energy path search* via the string method in 2D (translational)
  and 3D (roto-translational) configuration space (`flake string`), giving
  static friction forces and transition pathways.
- *Parameter-sweep Langevin dynamics* (`flake sweep`) parallelised over a
  parameter grid via joblib, for depinning thresholds and
  velocity–force scaling laws.
- *Substrate types*: sinusoidal (optical lattices), Gaussian well (ILP-like
  registry dependence), tanh well (lithographic patterns), and flat (testing
  baseline); new types require only a single `@njit` function.
- *Cluster geometries*: circle, hexagon, triangle, rectangle, ellipse, and
  arbitrary polygon via Shapely; multi-site bases supported.
- A *YAML-driven CLI* and six end-to-end Jupyter notebooks in `examples/`
  covering the full workflow from substrate construction to scaling-law
  analysis, with no Python scripting required.

# Research Impact

FLAKE v0.1.3 is the formalised, tested, and packaged release of a codebase
developed from 2022 onward at SISSA. The predecessor (v0.0.6, publicly
available since November 2022) was the primary simulation tool for the
following published studies:

- @silva_frictionless_2023 (Nanoscale): geometric classification of arbitrary 2D interfaces and scaling laws of static friction force from finite commensurate and incommensurate contacts.
- @cao_moire_2022 (Physical Review X): coupled roto-translational depinning and the role of moiré pattern evolution in colloidal experiments.
- @liao_twisting_2023 (ACS Applied Materials and Interfaces): combined experiment and simulation of twisting dynamics in large lattice-mismatch van der Waals heterostructures.

The colloidal experiments of @cao_orientational_2019 and @cao_pervasive_2021 provided the physics foundation and motivated the initial design of the package.

Version 0.1.3 adds a fully JIT-compiled MD inner loop, the roto-translational
string method, and a complete test suite. The package is currently in active use at SISSA for
ongoing research on quasicrystalline substrates and roto-translational
synchronisation effects, in collaboration with experimentalists in Germany
and China.

# AI Usage Disclosure

The physical model, all design decisions, and the research direction of FLAKE
are entirely the work of the author.
The central argument — that geometry dominates friction in stiff 2D contacts,
making a rigid-body model both sufficient and necessary for systematic study —
is grounded in the author's prior publications
[@silva_frictionless_2023; @cao_moire_2022].
The choice of substrate models, the roto-translational string method and its
arc-length metric, and the scope of the package all predate any AI involvement:
the core physical model and simulation algorithms were functional in the
predecessor codebase (v0.0.6, November 2022) prior to any AI assistance.
Claude Code (Anthropic, `claude-sonnet-4-6`) was subsequently used to improve
the codebase: making the code more
readable (docstring rewriting, module restructuring and API cleanup), more
efficient (Numba JIT refactoring of substrate closures and the MD inner loop),
and adherent to modern open-source standards (test suite scaffolding and CI
configuration) and improving this manuscript.
All AI-generated outputs were reviewed and edited by the author.

# Acknowledgements

The author thanks Emanuele Panizon, Jin Wang, Dong Han, Erio Tosatti, Andrea Vanossi,
Nicola Manini, Xin Cao, and Clemens Bechinger for the physics discussions and
experimental collaborations that motivated and shaped this work.
The author thanks Andrea Trost for help with the logo design.

This work was supported by the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation programme, grant
agreement No. 834402 (ULTRADISS), and by the Italian Ministry of University
and Research (MUR) under PRIN grant No. 20178PZCB5 (UTFROM).

# References
