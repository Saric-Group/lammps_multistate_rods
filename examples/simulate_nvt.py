# encoding: utf-8
'''
This application does a simple NVE+Langevin LAMMPS simulation of spherocylinder-like rods
(defined in a .cfg file) using the "lammps_multistate_rods" library.
The initial locations of the rods are at SC lattice points defined by the input params, and
their orientations are randomly determined at each insertion point.

Created on 16 Mar 2018

@author: Eugen Rožić
'''

import os
import argparse

parser = argparse.ArgumentParser(description='Program for NVE+Langevin hybrid LAMMPS simulation'\
                                 ' of spherocylinder-like rods, using the "lammps_multistate_rods"'\
                                 ' library.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('cfg_file',
                    help='path to the "lammps_multistate_rods" model configuration file')
parser.add_argument('run_file',
                    help='path to the run configuration file')
parser.add_argument('simlen', type=int,
                    help='the length of the simulation')

parser.add_argument('--seed', type=int,
                    help='the seed for random number generators')
parser.add_argument('--out', type=str, default=None,
                    help='name/path for the output folder (defaults to cfg_file path w/o ext)')

parser.add_argument('-o', '--output_freq', type=int,
                    help='configuration output frequency (in MD steps);'\
                    ' default behavior is after every batch of MC moves')
parser.add_argument('-s', '--silent', action='store_true',
                    help="doesn't print anything to stdout")

args = parser.parse_args()

if not args.cfg_file.endswith('.cfg'):
    raise Exception('Model configuration file (first arg) has to end with ".cfg"!')

if not args.run_file.endswith('.run'):
    raise Exception('Run configuration file (second arg) has to end with ".run"!')

if args.seed is None:
    import time
    seed = int((time.time() % 1)*1000000)
    print "WARNING: no seed given explicitly; using:", seed
else:
    seed = args.seed

if args.out is None:
    output_folder = os.path.splitext(args.cfg_file)[0]
else:
    output_folder = args.out

#========================================================================================

from lammps import PyLammps
import lammps_multistate_rods as rods

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

run_filename = os.path.splitext(os.path.basename(args.run_file))[0]
sim_ID = '{:s}_{:d}'.format(run_filename, seed)
    
dump_filename = sim_ID+'.dump'
dump_path = os.path.join(output_folder, dump_filename)

log_filename = '{:d}.lammps'.format(seed)
log_path = os.path.join(output_folder, log_filename)

run_args = rods.rod_model.Params()
execfile(args.run_file, {'__builtins__': None}, vars(run_args))

out_freq = args.output_freq if args.output_freq != None else run_args.mc_every

if args.silent:
    py_lmp = PyLammps()
else:
    py_lmp = PyLammps(cmdargs = ['-echo', 'both'])
py_lmp.log('"' + log_path + '"')
model = rods.Rod_model(args.cfg_file)
simulation = rods.Simulation(py_lmp, model, run_args.temp, seed, output_folder)

# SETUP AND CREATION
py_lmp.units("lj")
py_lmp.dimension(3)
py_lmp.boundary("p p p")
py_lmp.lattice("sc", 1/(run_args.cell_size**3))
py_lmp.region("box", "block", -run_args.num_cells / 2, run_args.num_cells / 2,
                              -run_args.num_cells / 2, run_args.num_cells / 2,
                              -run_args.num_cells / 2, run_args.num_cells / 2)
simulation.setup("box")
#simulation.create_rods() #same as "box = None"
# Sensible example for random creation:
overlap = (2.1 * model.rod_radius) / run_args.cell_size
maxtry = 10
simulation.create_rods(state_ID = 0, random = [int(run_args.num_cells**3), seed, "box",
                        "overlap", overlap, "maxtry", maxtry])
simulation.create_rods(state_ID = 1, random = [int(run_args.num_cells**3), 2*seed, "box",
                        "overlap", overlap, "maxtry", maxtry])

# DYNAMICS
py_lmp.fix("thermostat", "all", "langevin",
           run_args.temp, run_args.temp, run_args.damp, seed)#, "zero yes")
simulation.set_rod_dynamics("nve")

if model.num_states > 1:
    mc_tries = int(run_args.mc_tries * simulation.rods_count())
    simulation.set_state_transitions(run_args.mc_every, mc_tries)#, opt = ["full_energy"])
    
concentration = 0 #TODO ??
mc_exchange_tries = 10
#simulation.set_state_concentration(0, concentration, run_args.mc_every, mc_exchange_tries) 

# OUTPUT
#TODO state change accept rate, rod state ratio, ... (args.silent)
dump_elems = "id x y z type mol"
py_lmp.dump("dump_cmd", "all", "custom", out_freq, '"' + dump_path + '"', dump_elems)
py_lmp.dump_modify("dump_cmd", "sort id")
py_lmp.thermo_style("custom", "step atoms", "pe temp",
                    " ".join(["v_{}".format(group_var)
                              for group_var in simulation.state_group_vars]),
                    "f_{}[2]".format(simulation.state_trans_fix), # state change successes
                    "f_{}[1]".format(simulation.state_trans_fix)) # state change attempts
py_lmp.thermo(out_freq)

# RUN
py_lmp.neigh_modify("every 1 delay 1")    
py_lmp.timestep(run_args.dt)
py_lmp.run(args.simlen)
