#!/usr/bin/env python3

import sys, os, json, logging, multiprocessing, argparse
from argparse import RawTextHelpFormatter
import numpy as np
from time import time
from functools import partial
from flake.cluster import cluster_from_params, rotate
from flake.substrate import substrate_from_params, calc_matrices_bvect
from flake.string_method import PotentialPathAnalyt, Path

def static_barrier(pos, inputs, calc_en_f, name=None, log_propagate=True, debug=False):
    """Compute the transition state barrier of a rigid cluster between two points of the substrate.

    The barrier is computed using the string algorithm [DOI: 10.1063/1.2720838].

    Inputs:
    - pos: position of particles in rigid cluster
    - inputs: dictionary defining the syste. Must contain the start (p0) and end (p1) of the initial path. Optionally, the number of points in the string (Npt=100) and number of iterations (Nsteps=3000)

    Returns the positions of the relaxed path, along with energy, force and torque evaluated along it.
    """

    N = pos.shape[0]
    if name == None:
        name = 'sting_N_%i' % N
        out_fname = '/dev/null' # if no name is given, do not write the results to file, just return them.
    else:
        out_fname = '%s-N_%i.dat' % (name, N)

    #-------- SET UP LOGGER -------------
    # For this threads and children
    c_log = logging.getLogger(name)
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    log_format = logging.Formatter('[%(levelname)5s - %(funcName)10s] %(message)s')
    if not log_propagate:
        console = open('console-%s-N_%i.log' % (name, N), 'w')
        handler = logging.StreamHandler(console)
        handler.setFormatter(log_format)
        c_log.addHandler(handler)

    #-------- READ INPUTS -------------
    if type(inputs) == str: # Inputs passed as path to json file
        with open(inputs) as inj:
            inputs = json.load(inj)
    else:
        inputs = inputs.copy() # Copy it so multiple threads can access it
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    en_params = inputs['en_params']
    Nsteps = 3000 # Relax for these many steps
    if 'Nsteps' in inputs.keys(): Nsteps = inputs['Nsteps']
    Npt = 100 # number of subdivisions of the path connecting a and b
    if 'Npt' in inputs.keys(): Npt = inputs['Npt']
    fix_ends = False
    if 'fix_ends' in inputs.keys(): fix_ends = bool(inputs['fix_ends']) # are the given end of the path free to move?
    L = Path(inputs['p0'], inputs['p1'], pos, Npt, fix_ends=fix_ends) # initalise the path
    V = PotentialPathAnalyt(L, calc_en_f, en_params)  # potential along the path
    c_log.info("Relax string of %i points in %i stesp (%s endpoints)" % (Npt, Nsteps, 'fix' if fix_ends else 'free'))
    c_log.info("Intial string from p0=(%.5g, %.5g) p1=(%.5g, %.5g)" % (*inputs['p0'], *inputs['p1']))
    data = np.zeros((Npt, 6))

    #-------- OUTPUT SETUP -----------
    print_skip = 1 # Timesteps to skip between prints
    if 'print_skip' in inputs.keys(): print_skip = inputs['print_skip']
    printerr_skip = int(Nsteps/10)
    outstream = open(out_fname, 'w')
    # !! Labels and print_status data structures must be coherent !!
    num_space, indlab_space = 35, 2 # Width printed numerical values, Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['ly', 'e_pot', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='lx', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                        for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [lx, ly, e_pot, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- STRING MINIMISATION ----------------
    t0 = time() # Start clock
    c_log.propagate = False
    infoskip = 20
    for i in range(Nsteps):
        if i % int(Nsteps/infoskip) == 0: c_log.info("On it %i (%.2f%%)" % (i, i/Nsteps*100))
        L.eulerArc(V, dt=1e-6)
        V.update(L)

    for it, (lx, ly) in enumerate(zip(L.x, L.y)):
        e_pot, forces, torque = calc_en_f(pos+[lx, ly], [lx, ly], *en_params)
        data[it] = [lx, ly, e_pot, forces[0], forces[1], torque]
        print_status()
    outstream.close()

    c_log.propagate = True
    t_exec = time() - t0
    c_log.info("Done in %is (%imin)" % (t_exec, t_exec/60))
    return data, L, V

if __name__ == "__main__":
    t0 = time()

    #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Compute the energy barrier between two points using the string algorithm.

    Create a clusters of different sizes and compute the energy barrier along a path joining two given points and realxed according to the string algorithm.

    The parameters defining the cluster and substrate are given in a JSON file.
    The JSON input must contain all inputs to create the substrate and clusters as needed by substrate_from_params and cluster_from_params funcitons.
    Additionally, the JSON file may contain the number of itearions for relaxing the string and the number of points comprising the string, given in the keys 'Nsteps' (defaults=3000) and 'Npt' (defulat=100), respectively.
    The two points must be defined as 2D arrays by the keywords 'p0' and 'p1'. These two points maybe kept fixed by adding a boolean element 'fix_ends' and set it to true.

    The output is printed on files named '-N_<number of particles>.dat'.

    The code can run in parallel different sizes.
    """,
    formatter_class=RawTextHelpFormatter)
    # Positional arguments
    parser.add_argument('filename',
                        type=str,
                        help='JSON input file.')
    # Optional args
    parser.add_argument('--Ns',
                        dest='Ns', type=int, required=True, nargs=2,
                        help='Start and finish sizes (in lattice repetitions, valid for both directions).')
    parser.add_argument('--dN',
                        dest='dN', type=int, default=1,
                        help='size spaceing (defult=1)')
    parser.add_argument('--np',
                        dest='np', type=int, default=1,
                        help='number of sizes to run in parallel (defult=1)')
    parser.add_argument('--debug',
                        dest='debug', type=bool, default=False,
                        help='print debug informations')
    #-------------------------------------------------------------------------------
    # Initialize and check variables
    #-------------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])

    # -------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)
    if args.debug: c_log.setLevel(logging.DEBUG)

    # -------- INPUTS --------
    with open(args.filename) as inj:
        inputs = json.load(inj)

    N0, N1, dN = *args.Ns, args.dN
    c_log.info("From Nl %i to %i steps %i" % (N0, N1, dN))

    # -------- SUBSTRATE ENERGY --------
    _, calc_en_f, en_params = substrate_from_params(inputs)
    inputs['en_params'] = en_params
    if inputs['well_shape'] == 'sin': inputs['S'] = [[1,0], [0,1]] # Might not be define, just use cartesian
    else: _, inputs['S'] = calc_matrices_bvect(inputs['b1'], inputs['b2'])
    c_log.info("%s sub parms: " % inputs['well_shape'] + " ".join(["%s" % str(i) for i in en_params]))

    # ------ CLUSTER ----------
    Nl_range = range(N0, N1, dN) # Sizes to explore, in parallels
    pos_vector = []
    c_log.info("%s cluster parms: " % inputs['cluster_shape'] + " ".join(["%s" % str(i) for i in [inputs['a1'], inputs['a2'], inputs['cl_basis']]]))
    for Nl in Nl_range:
        inputs['N1'], inputs['N2'] = Nl, Nl
        pos = cluster_from_params(inputs)
        pos_vector.append(pos.copy()) # Just to be sure, pass a copy
        c_log.info("cluster size %i (Nl %i)" % (pos.shape[0], Nl))

    # ------------ MULTIPROCESS POOL ---------
    # Set up system for multiprocess
    ncpu, nworkers = os.cpu_count(), args.np
    c_log.info("Running %i elements on %i processes (%i cores machine)" % (len(Nl_range), nworkers, ncpu))

    # Fix the all arguments a part from Nl, so that we can use pool.map
    wrap_func = partial(static_barrier, inputs=inputs, calc_en_f=calc_en_f, name='barrier', log_propagate=False, debug=args.debug)

    # Launch all simulations with Pool of workers
    c_log.debug("Starting pool")
    pool = multiprocessing.Pool(processes=nworkers)
    results = pool.map(wrap_func, pos_vector)
    pool.close() # Pool doesn't accept new jobs
    pool.join() # Wait for all processes to finish
    c_log.debug("Results: %s" % str(results))

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin, %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
