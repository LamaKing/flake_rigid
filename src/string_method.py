"""
String method for minimum energy path (MEP) search in the translational plane.

Reference: E, Ren, Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007).

The simplified string method:
  1. Initialise N_pt points on a straight line from p0 to p1.
  2. Each point steps along the force: x_i += dt * F(x_i).
  3. Reparametrize to equal arc length (linear interpolation).
  4. Repeat until convergence or max_steps.

The dt parameter controls the step size.  For typical substrate
corrugation amplitudes of order epsilon ~ 1, dt ~ 1e-4 to 1e-5 is safe.
For much stronger corrugation, reduce dt to avoid overshooting.

Classes:  StringPath, StringPotential
Function: find_mep
"""

import numpy as np


class StringPath:
    """A discrete path in the 2D translational plane.

    Attributes:
        points:    (n_pt, 2) float64 ndarray -- current path coordinates.
        fix_ends:  bool -- if True, first and last points are frozen.
    """

    def __init__(self, p0, p1, n_pt, fix_ends=True):
        """Initialise as a straight line from p0 to p1.

        Args:
            p0:        (2,) array-like -- starting point.
            p1:        (2,) array-like -- ending point.
            n_pt:      int             -- number of points along path.
            fix_ends:  bool            -- freeze endpoints (default True).
        """
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        t  = np.linspace(0., 1., n_pt)
        # Linear interpolation: points[:,0] = p0[0] + t*(p1[0]-p0[0]), etc.
        self.points   = p0 + t[:, np.newaxis] * (p1 - p0)
        self.fix_ends = fix_ends

    @property
    def arc_length(self):
        """Total arc length of the current path."""
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def reparametrize(self):
        """Redistribute points to equal arc-length spacing (linear interp).

        Linear interpolation is preferred over cubic: cubic can overshoot
        for sharply curved paths near saddle points, introducing spurious
        oscillations that prevent convergence.

        Raises:
            RuntimeError: if the path has collapsed to a single point.
        """
        n_pt = len(self.points)
        diffs = np.diff(self.points, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        total   = seg_len.sum()

        if total < 1e-12:
            raise RuntimeError(
                "String collapsed to a point (total arc length < 1e-12)."
            )

        # Cumulative arc length parameter, normalised to [0, 1].
        s = np.concatenate([[0.], np.cumsum(seg_len)]) / total
        t_new = np.linspace(0., 1., n_pt)

        # Linear interpolation for x and y separately.
        new_x = np.interp(t_new, s, self.points[:, 0])
        new_y = np.interp(t_new, s, self.points[:, 1])
        self.points = np.column_stack([new_x, new_y])

    def step(self, forces, dt):
        """Move each point along the force direction, then reparametrize.

        Args:
            forces: (n_pt, 2) ndarray -- force F = -dV/dr at each point.
            dt:     float             -- step size.
        """
        forces = np.asarray(forces, dtype=np.float64)
        if self.fix_ends:
            forces[0]  = 0.
            forces[-1] = 0.
        self.points += dt * forces
        self.reparametrize()


class StringPotential:
    """Evaluate substrate energy and force along a path.

    Args:
        pos:       (N, 2) ndarray  -- cluster positions (fixed during MEP).
        calc_en_f: callable        -- total energy function from substrate_from_params.
        en_params: list            -- extra arguments for calc_en_f.
    """

    def __init__(self, pos, calc_en_f, en_params):
        self.pos       = np.asarray(pos, dtype=np.float64)
        self.calc_en_f = calc_en_f
        self.en_params = en_params

    def evaluate(self, path_points, n_jobs=1):
        """Compute energy and force at every point along the path.

        Args:
            path_points: (n_pt, 2) ndarray -- CM positions to evaluate.
            n_jobs:      int               -- joblib parallel workers.

        Returns:
            energies: (n_pt,) ndarray
            forces:   (n_pt, 2) ndarray  -- F = -dV/dr (from calc_en_f).
        """
        path_points = np.asarray(path_points, dtype=np.float64)
        n_pt = len(path_points)

        if n_jobs == 1:
            results = [
                self.calc_en_f(
                    self.pos + path_points[i], path_points[i], *self.en_params
                )
                for i in range(n_pt)
            ]
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "joblib is required for n_jobs != 1. "
                    "Install with: pip install joblib"
                )
            def _eval(i):
                cm = path_points[i]
                return self.calc_en_f(self.pos + cm, cm, *self.en_params)
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_eval)(i) for i in range(n_pt)
            )

        energies = np.array([r[0] for r in results])
        forces   = np.array([r[1] for r in results])
        return energies, forces


# ============================================================
# MEP finder
# ============================================================

def find_mep(pos, calc_en_f, en_params, p0, p1,
             n_pt=100, max_steps=3000, dt=1e-4,
             fix_ends=True, tol=1e-8, n_jobs=1):
    """Find the minimum energy path between p0 and p1.

    Args:
        pos:       (N, 2) ndarray  -- cluster positions (fixed).
        calc_en_f: callable        -- total energy function.
        en_params: list            -- extra arguments for calc_en_f.
        p0:        (2,) array-like -- start point (real-space CM).
        p1:        (2,) array-like -- end point.
        n_pt:      int             -- number of points along path.
        max_steps: int             -- maximum gradient-descent iterations.
        dt:        float           -- step size.  May need tuning for
                                     substrates with very different
                                     corrugation scales.
        fix_ends:  bool            -- freeze endpoints (default True).
        tol:       float           -- convergence: stop when max pointwise
                                     displacement between steps < tol.
        n_jobs:    int             -- parallel workers for potential eval.

    Returns:
        dict with keys:
            'points'    : (n_pt, 2)  -- final path in real space.
            'energy'    : (n_pt,)    -- energy along final path.
            'force'     : (n_pt, 2)  -- force along final path.
            'converged' : bool
            'n_steps'   : int        -- actual iterations taken.
            'barrier'   : float      -- max(energy) - min(energy).
    """
    path      = StringPath(p0, p1, n_pt, fix_ends=fix_ends)
    potential = StringPotential(pos, calc_en_f, en_params)

    converged = False
    step_count = 0

    for step_count in range(1, max_steps + 1):
        prev_points = path.points.copy()
        _, forces   = potential.evaluate(path.points, n_jobs=n_jobs)
        path.step(forces, dt)

        # Max displacement of any point since last step (after reparametrization).
        max_disp = np.max(np.linalg.norm(path.points - prev_points, axis=1))
        if max_disp < tol:
            converged = True
            break

    energies, forces = potential.evaluate(path.points, n_jobs=n_jobs)

    return {
        'points':    path.points.copy(),
        'energy':    energies,
        'force':     forces,
        'converged': converged,
        'n_steps':   step_count,
        'barrier':   float(energies.max() - energies.min()),
    }
