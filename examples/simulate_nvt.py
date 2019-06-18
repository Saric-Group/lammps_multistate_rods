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

#from mpi4py import MPI #TODO make MPI work...
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

out_freq = args.output_freq if args.output_freq != None else run_args.run_length

py_lmp = PyLammps(cmdargs=['-screen','none'])
py_lmp.log('"'+log_path+'"')
model = rods.Rod_model(args.cfg_file)
simulation = rods.Simulation(py_lmp, model, seed, output_folder)

py_lmp.units("lj")
py_lmp.dimension(3)
py_lmp.boundary("p p p")
py_lmp.lattice("sc", 1/(run_args.cell_size**3))
py_lmp.region("box", "block", -run_args.num_cells / 2, run_args.num_cells / 2,
                              -run_args.num_cells / 2, run_args.num_cells / 2,
                              -run_args.num_cells / 2, run_args.num_cells / 2)
simulation.setup("box")
simulation.create_rods(box = None)

# DYNAMICS
py_lmp.fix("thermostat", "all", "langevin",
           run_args.temp, run_args.temp, run_args.damp, seed)#, "zero yes")
simulation.set_rod_dynamics("nve")

py_lmp.neigh_modify("every 1 delay 1")

# GROUPS & COMPUTES
if hasattr(run_args, 'micelle_cutoff') and run_args.micelle_cutoff > 0.0:
    micelle_group = 'active_sol_beads'
    py_lmp.variable('active_sol', 'atom', '"' + 
                    ' || '.join(['(type == {:d})'.format(t) for t in
                                    filter(lambda t: t in model.state_bead_types[0],
                                           model.active_bead_types)])
                    + '"')
    py_lmp.group(micelle_group, 'dynamic', simulation.rods_group,
                 'var', 'active_sol', 'every', out_freq)
    micelle_compute = "rod_micelle"
    py_lmp.compute(micelle_compute, micelle_group, 'aggregate/atom', run_args.micelle_cutoff)
    
if hasattr(run_args, 'fibril_cutoff') and run_args.fibril_cutoff > 0.0:
    fibril_group = 'active_beta_beads'
    py_lmp.variable('active_beta', 'atom', '"' + 
                    ' || '.join(['(type == {:d})'.format(t) for t in
                                    filter(lambda t: t in model.state_bead_types[1],
                                           model.active_bead_types)])
                    + '"')
    py_lmp.group(fibril_group, 'dynamic', simulation.rods_group,
                 'var', 'active_beta', 'every', out_freq)
    fibril_compute = "rod_fibril"
    py_lmp.compute(fibril_compute, fibril_group, 'aggregate/atom', run_args.fibril_cutoff)

# OUTPUT
py_lmp.thermo_style("custom", "step atoms", "pe temp")
dump_elems = "id x y z type mol"
try:
    dump_elems += " c_"+micelle_compute
except:
    pass #isn't defined
try:
    dump_elems += " c_"+fibril_compute
except:
    pass #isn't defined
py_lmp.dump("dump_cmd", "all", "custom", out_freq, dump_path, dump_elems)
py_lmp.dump_modify("dump_cmd", "sort id")
py_lmp.thermo(out_freq)

# RUN...
mc_moves_per_run = 0
if model.num_states > 1:
    mc_moves_per_run = int(run_args.mc_moves * simulation.rods_count())
    
py_lmp.timestep(run_args.dt)

if mc_moves_per_run == 0:
    py_lmp.command('run {:d}'.format(args.simlen))
else:
    py_lmp.command('run {:d} post no'.format(run_args.run_length-1)) #so output happens after state changes
    remaining = args.simlen - run_args.run_length + 1
    for i in range(args.simlen / run_args.run_length):
        success = simulation.state_change_MC(mc_moves_per_run)#, replenish=("box", 2*model.rod_radius, 10)) TODO
        if not args.silent:
            base_count = simulation.state_count(0)
            beta_count = simulation.state_count(1)
            print 'step {:d} / {:d} :  beta-to-soluble ratio = {:d}/{:d} = {:.5f} (accept rate = {:.5f})'.format(
                    (i+1)*run_args.run_length, args.simlen, beta_count, base_count, 
                    float(beta_count)/base_count, float(success)/mc_moves_per_run)
        
        if remaining / run_args.run_length > 0:
            py_lmp.command('run {:d} post no'.format(run_args.run_length))
            remaining -= run_args.run_length
        else:
            py_lmp.command('run {:d} post no'.format(remaining))
