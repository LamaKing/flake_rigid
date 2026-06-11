"""
FLAKE -- Friction and Lattice Analysis of Kinetics and Energetics.

Overdamped Langevin MD of rigid 2D clusters on periodic substrates.

Modules
-------
    flake.substrate    -- substrate energy functions (sin, gaussian, tanh, flat)
    flake.cluster      -- cluster geometry and Langevin coefficients
    flake.dynamics     -- Euler-Maruyama integrator (run_md)
    flake.sweep        -- parameter sweeps over MD runs
    flake.maps         -- translational / rotational / rototrasl energy maps
    flake.string_method -- minimum energy path (string method)
    flake.io           -- HDF5 / YAML trajectory I/O
    flake.plot         -- plotting utilities
    flake.cli          -- command-line interface
"""
