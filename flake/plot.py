import numpy as np
import matplotlib.pyplot as plt
import os, shutil, json, sys
from os.path import join as pjoin

from monty.json import MSONable, MontyEncoder, MontyDecoder, MSONError

def plot_UC(ax, u, params={'ls': ':', 'color': 'tab:gray', 'lw': 1}):
    """Shortcut to plot the unit cell of the lattice"""
    BZ_corner = np.array([(n*u[0]+m*u[1])
                          for n,m in [[0,0], [1,0], [1,1], [0,1], [0,0]]
                         ])
    for i in range(4):
        ax.plot([BZ_corner[i+1,0], BZ_corner[i,0]],
                [BZ_corner[i+1,1], BZ_corner[i,1]], **params)
    return ax

def get_brillouin_zone_2d(cell):
    """
    Generate the Brillouin Zone of a given 2D cell. The BZ is the Wigner-Seitz cell
    of the reciprocal lattice, which can be constructed by Voronoi decomposition
    to the reciprocal lattice.  A Voronoi diagram is a subdivision of the space
    into the nearest neighborhoods of a given set of points.

    See:
    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams
    http://staff.ustc.edu.cn/~zqj/posts/howto-plot-brillouin-zone/

    Returns list of vertices of the cell.
    """

    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (2, 2)

    # Compute nearest neighbours
    nn = np.array([i*cell[0]+j*cell[1] for j in range(-1, 2) for i in range(-1,2)])

    from scipy.spatial import Voronoi
    vor = Voronoi(nn)
    # Get the region to which the origin belongs to (origin is index 4, i,j=(0,0))
    orig_region = vor.regions[vor.point_region[4]]
    verts = vor.vertices[orig_region] # Vertices of the corresponding regionx
    return verts

def plot_BZ2d(ax, ws_verts, params={'ls': '--', 'color': 'tab:gray', 'lw': 1, 'fill': False}):
    """Create a patch of the Brillouin Zone and add to MPL axis.

    Inputs are a matplolib axis to plot (e.g. plt.gca()) and list of vertices (e.g. computed by get_brillouin_zone_2d).
    Optionally dictionary with plotting parameters.

    Returns the axis and cell Polygon object
    """

    from matplotlib.patches import Polygon
    ws_cell = Polygon(ws_verts, **params)
    ax.add_patch(ws_cell)
    ax.set_aspect('equal') # I can't think of a situation where you would want arbitrary distorion on the shape.
    return ax, ws_cell

def plt_cosmetic(ax, xlabel='x', ylabel='y'):
    """Adjust ax for plotting: lines at x=0 y=0, set xylabels and aspect as equal"""
    ax.axhline(color='gray', ls=':', lw=1)
    ax.axvline(color='gray', ls=':', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')

def handle_run(params, c_key, c_val, driver, move_fname=[], outfname='out.dat'):
    """Change param file according to given key and value.
    Run the driver.
    Create a folder "key_value" according to key and value.
    Move parameter and output files in the folder.
    Return name of the folder."""

    if type(c_key)==list:
        if len(c_key) != len(c_val): raise ValueError('Length key (%i) and value (%i) do not match' % (len(c_key), len(c_val)))
    else: c_key, c_val = [c_key], [c_val]
    cdir = ''
    for cc_key, cc_val in zip(c_key, c_val):
        params[cc_key] = cc_val
    cdir = '-'.join(['%s_%.4g' % (cc_key, cc_val) for cc_key, cc_val in zip(c_key, c_val)])
    os.makedirs(cdir, exist_ok=True)

    pwd =  os.environ['PWD']
    print('Working in ', pwd)

    if outfname not in move_fname: move_fname += [outfname]
    print('moving', move_fname)
    # Run
    with open(outfname, 'w') as outstream:
        driver(params, name='', outstream=outstream)

    for cfname in move_fname:
        shutil.move(pjoin(pwd, cfname), pjoin(pwd, cdir, cfname))
    # Copy input in folder
    with open(pjoin(pwd, cdir, 'params.json'), 'w') as outj:
        #json.dump(params, outj, indent=4)
        json.dump(params, outj, indent=4, cls=MontyEncoder)

    return cdir

def plot_lattice_vectors(ax, S, colors=('tab:red', 'tab:orange'), labels=('b1', 'b2')):
    """Plot the two primitive lattice vectors as arrows from the origin."""
    for vec, color, label in zip(S, colors, labels):
        ax.quiver(0, 0, vec[0], vec[1],
                  angles='xy', scale_units='xy', scale=1,
                  zorder=5, color=color, label=label)
    return ax
