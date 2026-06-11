#!/usr/bin/env python3

import sys, os, json, logging, argparse
from argparse import RawTextHelpFormatter
import numpy as np
from time import time
from flake.cluster import create_cluster, load_cluster, rotate, calc_cluster_langevin
from flake.substrate import substrate_from_params

from monty.json import MSONable, MontyEncoder, MontyDecoder, MSONError

def MD_rigid(inputs, outstream=sys.stdout, name=None, log_propagate=False, debug=False):
    """Overdamped (1st order differential equaltion) Langevin Molecular Dynamics of rigid cluster over a substrate.

    Input parameters are passaed as directly as dictionary or read from a JSON file (if inputs is a string).
    Position of particles in cluster is given as .hex or .npy filename.
    Info about run are written in name+'info.json'

    MD output is printed on sys stdout.
    """

    t0 = time() # Start clock

    if name == None: name = 'MD_rigid'

    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger(name) # Set name of the function
    logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)

    #-------- INPUTS -------------
    if type(inputs) == dict: # Inputs passed as python dictionary
        inputs = inputs
    elif type(inputs[0]) == str: # Inputs passed as path to json file
        with open(inputs[0]) as inj:
            #inputs = json.load(inj)
            from monty.json import MSONable, MontyEncoder, MontyDecoder, MSONError
            inputs = json.load(inj, cls=MontyDecoder)
    else:
        raise TypeError('Unrecognized input structure (no dict or filename str)', inputs)
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    #-------- CLUSTER --
    pos_cm = np.zeros(2, dtype=float) # If not given, start from centre
    theta = 0 # If not given, start aligned
    if 'theta' in inputs.keys(): theta = inputs['theta'] # Starting angle [deg]
    if 'pos_cm' in inputs.keys(): pos_cm = np.array(inputs['pos_cm'], dtype=float) # Start pos [micron]
    # Load
    if 'cluster_hex' in inputs.keys():
        input_cluster = inputs['cluster_hex'] # Geom as lattice and Bravais points
        pos = create_cluster(input_cluster, theta)[:,:2]
    elif 'cluster_np' in inputs.keys():
        input_cluster = inputs['cluster_np'] # Geom as numpy datfile
        pos = load_cluster(input_cluster, theta)[:,:2]
    else: raise KeyError("No input for cluster (hex or np)")

    test_cm = np.mean(pos, axis=0)
    if not np.isclose(np.linalg.norm(test_cm - pos_cm), 0):
        c_log.warning('Pos CM (%s) and input CM (%s) are different' % (str(test_cm), str(pos_cm)))
    N = pos.shape[0] # Size of the cluster
    c_log.info("Cluster N=%i start at (x,y)=(%.3g,%.3g), theta=%.3g)" % (N, *pos_cm, theta))

    #-------- SUBSTRATE ENERGY --------
    _, calc_en_f, en_params = substrate_from_params(inputs) # ignore per-particle function
    c_log.info("%s sub parms: " % inputs['well_shape'] + " ".join(["%s" % str(i) for i in en_params]))

    #-------- MD params --
    T = inputs['T'] # kBT [zJ]
    Tau = inputs['Tau'] # [fN*micron]
    Fx, Fy = inputs['Fx'], inputs['Fy'] # [fN]
    F = np.array([Fx, Fy])
    Nsteps = inputs['Nsteps']
    dt = inputs['dt'] # [ms]

    print_skip = 100 # Timesteps to skip between prints
    if 'print_skip' in inputs.keys(): print_skip = inputs['print_skip']
    printprog_skip = int(Nsteps/5) # Progress output frequency
    c_log.debug("Print every %i timesteps. Status update every %i." % (print_skip, printprog_skip))

    # initialise variable
    forces, torque = np.zeros(2), 0.
    #------------------------

    ###############################
    #  STOPPING CONDITIONS
    both_breaks = True # Break if both V and omega satisfy conditions
    if 'both_breaks' in inputs.keys(): both_breaks = bool(inputs['both_breaks'])
    break_omega, break_V = False, False
    omega_avg, vel_avg = 0, 0 # store average of omega and velox over given timesteps
    avglen = 100 # timesteps for average
    min_Nsteps = 1e30 # min steps
    omega_min, omega_max = -1, 1e30 # tolerance (>0) to consider the system stuck or depinned.
    vel_min, vel_max = -1, 1e30     # If not given, continue indefinitely: max huge, min <0
    rcm_max, theta_max = 1e30, 1e30

    # Set Stuck config exit
    if 'min_Nsteps' in inputs.keys(): min_Nsteps = inputs['min_Nsteps']
    if 'omega_min' in inputs.keys(): omega_min = inputs['omega_min']
    if 'vel_min' in inputs.keys(): vel_min = inputs['vel_min']

    c_log.debug("Stuck %s: Nmin=%g (tmin=%g) avglen %i omega_min=%.4g deg/ms velox_min=%.4g micron/ms" % ('both' if both_breaks else 'single', min_Nsteps, min_Nsteps*dt, avglen, omega_min, vel_min))
    if min_Nsteps < avglen: raise ValueError("Omega/Velocity average length larger them minimum number of steps!")

    # Set pinning config exit
    if 'theta_max' in inputs.keys(): theta_max = inputs['theta_max']+theta # Exit if cluster rotates more than this
    if 'rcm_max' in inputs.keys(): rcm_max = inputs['rcm_max']+np.linalg.norm(pos_cm) # Exit if cluster slides more than this
    if 'omega_max' in inputs.keys(): omega_max = inputs['omega_max'] # Exit if cluster rotates faster than this
    if 'vel_max' in inputs.keys(): vel_max = inputs['vel_max'] # Exit if cluster rotates more than this
    c_log.debug("Depin: theta_max=%.4g omega_max=%.4g rcm_max=%.4g vel_max = %.4g" % (theta_max, rcm_max, omega_max, vel_max))
    ###############################

    #-------- LANGEVIN ----------------
    # Assumes rotation and translation indipendent. We just care about the scaling, not exact number.
    eta = 1   # Translational damping of single colloid
    if 'eta' in inputs.keys(): eta = inputs['eta']
    # CM translational viscosity [fKg/ms], CM rotational viscosity [micron^2*fKg/ms]
    etat_eff, etar_eff = calc_cluster_langevin(eta, pos)
    # Aplitude of random numbers. Translational follows nicely from CLT, rotational is assumed from A. E. Filippov, M. Dienwiebel, J. W. M. Frenken, J. Klafter, and M. Urbakh, Phys. Rev. Lett. 100, 046102 (2008).
    brandt, brandr = np.sqrt(2*T*etat_eff), np.sqrt(2*T*etar_eff)
    kBTroom = 4.069767441860465 #zJ
    seed = 12345
    if 'seed' in inputs.keys(): seed = inputs['seed']
    np.random.seed(seed)
    c_log.info("Random noise from seed %i" % seed)
    c_log.info("Number of particles %i Eta trasl %.5g Eta tras eff %.5g Eta roto eff %.5g Ratio roto/tras %.5g kBT=%.3g (kBT/kBTroom=%.3g)" % (N, eta, etat_eff, etar_eff, etar_eff/etat_eff, T, T/kBTroom))

    c_log.info("Tau = %.4g fN*micron (Tau/N=%.4g)" % (Tau, Tau/N))
    c_log.info("Fx=%.4g fN (Fx/N=%.4g), Fy=%.4g fN (Fy/N=%.4g), |F| = %.4g fN (|F|/N=%.4g)",
               Fx, Fx/N, Fy, Fy/N, np.sqrt(Fx**2+Fy**2), np.sqrt(Fx**2+Fy**2)/N)
    c_log.info("Omega free=%.4g  Vfree=(%.4g,%.4g) |Vfree|=%.4g" % (Tau/etar_eff, Fx/etat_eff, Fy/etat_eff, np.sqrt(Fx**2+Fy**2)/etat_eff))
    c_log.debug("Free cluster would rotate %.2f deg", Tau/etar_eff * Nsteps * dt)
    c_log.debug("Free cluster would translate %.2f micron", np.sqrt(Fx**2+Fy**2)/etat_eff * Nsteps * dt)
    c_log.debug("Amplitude of random number trasl %.2g and roto %.2g" % (brandt, brandr))
    #------------------------

    #-------- INFO FILE ---------------
    with open(name+'info.json', 'w') as infof:
        infod = {'eta': eta, 'etat_eff': etat_eff, 'etar_eff': etar_eff, 'brandt': brandt, 'brandr': brandr,
                 'N': N, 'theta_max': theta_max, 'print_skip': print_skip,
                 'min_Nsteps': min_Nsteps, 'avglen': avglen, 'omega_min': omega_min, 'omega_max': omega_max,
                 'pos_cm': pos_cm.tolist()
        }
        infod.update(inputs)
        #json.dump(infod, infof, indent=True, cls=MontyEncoder)
    #------------------------

    #-------- OUTPUT SETUP -----------
    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['e_pot', 'pos_cm[0]', 'pos_cm[1]', 'Vcm[0]', 'Vcm[1]',
                     'theta', 'omega', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='t', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                       for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [dt*it, e_pot, pos_cm[0], pos_cm[1], Vcm[0], Vcm[1],
                theta, omega, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)
    #------------------------

    #-------- START MD ----------------
    t0 = time() # Start clock
    for it in range(Nsteps):

        # ENERGY LANDSCAPE
        e_pot, forces, torque = calc_en_f(pos + pos_cm, pos_cm, *en_params)

        # UPDATE VELOCITIES
        # First order Langevin equation
        noise = np.random.normal(0, 1, size=3)
        Vcm = (forces + F + brandt*noise[0:2])/etat_eff
        omega = (torque + Tau + brandr*noise[2])/etar_eff

        # Print progress
        if it % printprog_skip == 0:
            c_log.info("t=%8.3g of %5.2g (%2i%%) E=%10.3g  x=%9.3g y=%9.3g theta=%9.3g omega=%10.3g |Vcm|=%8.3g",
                       it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], theta, omega, np.linalg.norm(Vcm))
            c_log.debug("Break: v %.5g (vmin %.5g vmax %.5g); omega %.5g (omegamin %.5g omegamax %.5g)" % (np.linalg.norm(Vcm), vel_min, vel_max, omega, omega_min, omega_max))
        # Print step results
        if it % print_skip == 0: print_status()

        # UPDATE DEGREES OF FREEDOM
        pos_cm += dt * Vcm # center of mass follows local forces.
        theta += dt*omega # theta of cluster follows local torque.
        # positions are further rotated
        pos = rotate(pos, dt*omega)

        ###############################
        # CHECK FOR STOPPING CONDITIONS
        # Compute omega average and check for exit conditions
        omega_avg += omega # Average omega to check if system is stuck. See avglen.
        vel_avg += np.linalg.norm(Vcm) # Average omega to check if system is stuck. See avglen.
        if it % avglen == 0:
            omega_avg /= avglen
            vel_avg /= avglen
            rcm = np.linalg.norm(pos_cm)

            # If system is stuck, set flag to exit
            if np.abs(omega_avg) < omega_min and it >= min_Nsteps: break_omega = True
            if vel_avg < vel_min and it >= min_Nsteps: break_V = True

            # If system is rotating or sliding without stopping, set flag to exit
            if (theta >= theta_max or np.abs(omega_avg) >= omega_max) and it >= min_Nsteps: break_omega = True
            if (rcm >= rcm_max or vel_avg >= vel_max) and it >= min_Nsteps: break_V = True

            # Break if either or both condistions are satisfied
            if ((break_omega and break_V) and both_breaks) or ((break_omega or break_V) and not both_breaks):
                # Values
                c_log.info("Rotational: theta=%10.4g <omega>_%i=%10.4g" % (theta, avglen, omega_avg))
                c_log.info("Translationa: rcm=%10.4g <v>_%i=%10.4g" % (rcm, avglen, vel_avg))
                # Pinned check
                if np.abs(omega_avg) < omega_min: c_log.info("System is roto-pinned (omega min=%.4g)." % (omega_min))
                if np.abs(vel_avg) < vel_min: c_log.info("System is trasl-pinned (vel_min=%.4g)." % (vel_min))
                # Depinned check
                if (theta >= theta_max or np.abs(omega_avg) >= omega_max):
                    c_log.info("System is roto-depinned (theta max=%10.4g omega max=%10.4g)" % (theta_max, omega_max))
                if (rcm >= rcm_max or vel_avg >= vel_max):
                    c_log.info("System is trasl-depinned (rcm max=%10.4g vel max=%10.4g)" % (rcm_max, vel_max))
                # Exit conditions
                c_log.info("Breaking condition (%s) satisfied. Exit" % ('both omega and Vcm' if both_breaks else 'single'))
                break

            omega_avg = 0 # Reset average
            vel_avg = 0
        ###############################
    #-----------------------

    # Print last step
    c_log.info("t=%8.3g of %5.2g (%2i%%) E=%10.3g  x=%9.3g y=%9.3g theta=%9.3g omega=%10.3g |Vcm|=%8.3g",
               it*dt, Nsteps*dt, 100.*it/Nsteps, e_pot, pos_cm[0], pos_cm[1], theta, omega, np.linalg.norm(Vcm))
    print_status()

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin or %.2fh). Speed %5.3f s/step" % (t_exec, t_exec/60, t_exec/3600, t_exec/Nsteps))

# Stand-alone scripting
if __name__ == "__main__":
    #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Run an MD simulations for a given cluster
    """,
    formatter_class=RawTextHelpFormatter)
    # Positional arguments
    parser.add_argument('filename',
                        type=str,
                        help='JSON input file.')
    parser.add_argument('--debug',
                        dest='debug', type=bool, default=False,
                        help='print debug informations')
    #-------------------------------------------------------------------------------
    # Initialize and check variables
    #-------------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])

    MD_rigid([args.filename], debug=args.debug)
