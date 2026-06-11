"""
String method for minimum energy path (MEP) search.

Two usage modes
---------------
2D translational (x_cm, y_cm) -- fixed orientation:
    Caller pre-rotates the cluster to the desired angle before calling.

        from flake.cluster import rotate
        pos_rot = rotate(pos, theta_deg)
        result  = find_mep(pos_rot, calc_en_f, en_params,
                           p0=[x0, y0], p1=[x1, y1])

    result['points'] has shape (n_pt, 2).

3D roto-translational (x_cm, y_cm, theta_deg):
    pos is the cluster in the reference (theta=0) frame.  The string
    method rotates pos internally at each path point.

        result = find_mep(pos, calc_en_f, en_params,
                          p0=[x0, y0, th0], p1=[x1, y1, th1],
                          scale=[lambda_x, lambda_y, lambda_theta])

    result['points'] has shape (n_pt, 3).  theta is in degrees.
    Typical scale values: lambda_x = lambda_y = substrate lattice
    spacing; lambda_theta = angular period (e.g. 60 deg for a 6-fold
    symmetric contact, 30 deg for 12-fold).

Gradients and string step
--------------------------
The gradient of E with respect to path coordinates:
    -dE/dx  = Fx,   -dE/dy  = Fy,   -dE/dtheta = tau

The string step moves each interior point along the gradient of -E:
    delta_p = (Fx, Fy)        for 2D
    delta_p = (Fx, Fy, tau)   for 3D

Scale parameter
---------------
scale affects ONLY the arc-length reparametrization, not the gradient
step.  It ensures that coordinates with different physical units (e.g.
Angstrom and degrees) contribute equally to arc length:
    ds^2 = sum_i (dp_i / lambda_i)^2

Pass scale=None (default) for 2D: all coordinates share the same units.

Algorithm
---------
Reference: E, Ren, Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007).

  1. Initialise n_pt points on a straight line from p0 to p1.
  2. Each interior point steps along the gradient: p_i += dt * grad_i.
  3. Reparametrize to equal arc-length (scaled, linear interpolation).
  4. Repeat until convergence or max_steps.

Classes:  StringPath, StringPotential
Function: find_mep
"""

import numpy as np


class StringPath:
    """A discrete path in dim-dimensional configuration space.

    Attributes:
        points:   (n_pt, dim) float64 ndarray -- current path coordinates.
        fix_ends: bool  -- if True, endpoints are frozen during steps.
        scale:    (dim,) float64 ndarray -- coordinate scales for arc length.
    """

    def __init__(self, p0, p1, n_pt, fix_ends=True, scale=None):
        """Initialise as a straight line from p0 to p1.

        Args:
            p0:       (dim,) array-like -- starting point.
            p1:       (dim,) array-like -- ending point.
            n_pt:     int               -- number of points along path.
            fix_ends: bool              -- freeze endpoints (default True).
            scale:    (dim,) array-like or None -- coordinate scales used
                                                   in arc-length metric.
                                                   Default: ones(dim).
        """
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        dim = p0.shape[0]

        t = np.linspace(0., 1., n_pt)
        self.points   = p0 + t[:, np.newaxis] * (p1 - p0)
        self.fix_ends = fix_ends
        self.scale    = (np.ones(dim, dtype=np.float64)
                         if scale is None
                         else np.asarray(scale, dtype=np.float64))

    @property
    def arc_length(self):
        """Total arc length of the path in the scaled metric."""
        scaled  = self.points / self.scale
        diffs   = np.diff(scaled, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def reparametrize(self):
        """Redistribute points to equal arc-length spacing (scaled metric).

        Linear interpolation is used to avoid cubic overshoot near
        saddle points, which can prevent convergence.

        The scale is applied only to arc-length computation; interpolation
        is performed on the unscaled coordinates.

        Raises:
            RuntimeError: if the path has collapsed to a point.
        """
        n_pt = len(self.points)

        scaled  = self.points / self.scale
        diffs   = np.diff(scaled, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        total   = seg_len.sum()

        if total < 1e-12:
            raise RuntimeError(
                "String collapsed to a point (arc length < 1e-12)."
            )

        s     = np.concatenate([[0.], np.cumsum(seg_len)]) / total
        t_new = np.linspace(0., 1., n_pt)

        new_points = np.empty_like(self.points)
        for d in range(self.points.shape[1]):
            new_points[:, d] = np.interp(t_new, s, self.points[:, d])
        self.points = new_points

    def step(self, gradients, dt):
        """Move each interior point along the gradient, then reparametrize.

        Args:
            gradients: (n_pt, dim) ndarray -- -dE/dp at each point.
            dt:        float               -- step size.
        """
        gradients = np.asarray(gradients, dtype=np.float64)
        if self.fix_ends:
            gradients[0]  = 0.
            gradients[-1] = 0.
        self.points += dt * gradients
        self.reparametrize()


class StringPotential:
    """Evaluate substrate energy and gradient along a path.

    Args:
        pos:       (N, 2) ndarray  -- cluster positions (reference frame, theta=0).
        calc_en_f: callable        -- total energy function from substrate_from_params.
        en_params: list            -- extra arguments for calc_en_f.
    """

    def __init__(self, pos, calc_en_f, en_params):
        self.pos       = np.asarray(pos, dtype=np.float64)
        self.calc_en_f = calc_en_f
        self.en_params = en_params

    def evaluate(self, path_points, n_jobs=1):
        """Compute energy and gradient (-dE/dp) at every point along the path.

        The dimension of path_points determines the evaluation mode:
          dim=2: translational (x_cm, y_cm); pos already oriented by caller.
          dim=3: roto-translational (x_cm, y_cm, theta_deg); pos rotated here.

        Args:
            path_points: (n_pt, dim) ndarray -- path coordinates.
            n_jobs:      int                 -- joblib parallel workers.

        Returns:
            energies:  (n_pt,)     ndarray
            gradients: (n_pt, dim) ndarray  -- (-dE/dx, -dE/dy) for dim=2;
                                               (-dE/dx, -dE/dy, -dE/dtheta)
                                               = (Fx, Fy, tau) for dim=3.
        """
        path_points = np.asarray(path_points, dtype=np.float64)
        dim = path_points.shape[1]

        if dim == 2:
            def _eval_point(cm):
                e, f, tau = self.calc_en_f(
                    self.pos + cm, cm, *self.en_params)
                return e, np.array([f[0], f[1]])

        elif dim == 3:
            from flake.cluster import rotate
            def _eval_point(pt):
                cm      = pt[:2]
                pos_rot = rotate(self.pos, float(pt[2]))
                e, f, tau = self.calc_en_f(
                    pos_rot + cm, cm, *self.en_params)
                return e, np.array([f[0], f[1], float(tau)])

        else:
            raise ValueError(
                "path_points must have dim 2 or 3, got %d" % dim
            )

        n_pt = len(path_points)

        if n_jobs == 1:
            results = [_eval_point(path_points[i]) for i in range(n_pt)]
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "joblib is required for n_jobs != 1. "
                    "Install with: pip install joblib"
                )
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_eval_point)(path_points[i]) for i in range(n_pt)
            )

        energies  = np.array([r[0] for r in results])
        gradients = np.array([r[1] for r in results])
        return energies, gradients


# ============================================================
# MEP finder
# ============================================================

def find_mep(pos, calc_en_f, en_params, p0, p1,
             n_pt=100, max_steps=3000, dt=1e-4,
             fix_ends=True, tol=1e-8,
             scale=None, n_jobs=1):
    """Find the minimum energy path between p0 and p1.

    The dimension of the search is inferred from len(p0):
      dim=2: translational (x_cm, y_cm); caller pre-rotates pos if theta != 0.
      dim=3: roto-translational (x_cm, y_cm, theta_deg); pos is the reference
             frame (theta=0); rotation is applied internally at each path point.

    Args:
        pos:       (N, 2) ndarray  -- cluster positions (reference frame).
        calc_en_f: callable        -- total energy function.
        en_params: list            -- extra arguments for calc_en_f.
        p0:        (dim,) array-like -- start point.
        p1:        (dim,) array-like -- end point.
        n_pt:      int             -- number of points along path.
        max_steps: int             -- maximum gradient-descent iterations.
        dt:        float           -- step size.
        fix_ends:  bool            -- freeze endpoints (default True).
        tol:       float           -- convergence: max pointwise displacement
                                      between iterations < tol.
        scale:     (dim,) array-like or None -- coordinate scales for the
                                      arc-length reparametrization.
                                      Default None (ones).
        n_jobs:    int             -- parallel workers for potential eval.

    Returns:
        dict with keys:
            'points'   : (n_pt, dim) -- final path in configuration space.
            'energy'   : (n_pt,)     -- energy along final path.
            'gradient' : (n_pt, dim) -- -dE/dp along final path.
            'converged': bool
            'n_steps'  : int         -- actual iterations taken.
            'barrier'  : float       -- max(energy) - min(energy).
            'dim'      : int         -- 2 or 3.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dim = p0.shape[0]

    path      = StringPath(p0, p1, n_pt, fix_ends=fix_ends, scale=scale)
    potential = StringPotential(pos, calc_en_f, en_params)

    converged  = False
    step_count = 0

    for step_count in range(1, max_steps + 1):
        prev_points       = path.points.copy()
        _, gradients      = potential.evaluate(path.points, n_jobs=n_jobs)
        path.step(gradients, dt)

        max_disp = np.max(np.linalg.norm(path.points - prev_points, axis=1))
        if max_disp < tol:
            converged = True
            break

    energies, gradients = potential.evaluate(path.points, n_jobs=n_jobs)

    return {
        'points':    path.points.copy(),
        'energy':    energies,
        'gradient':  gradients,
        'converged': converged,
        'n_steps':   step_count,
        'barrier':   float(energies.max() - energies.min()),
        'dim':       dim,
    }
