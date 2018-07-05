# encoding: utf-8
'''
This application does a simple NVE+Langevin LAMMPS simulation of spherocylinder-like rods
(defined in a .cfg file) using the "lammps_multistate_rods" library.

Created on 16 Mar 2018

@author: Eugen Rožić
'''

import argparse

parser = argparse.ArgumentParser(description='''Program for NVE+Langevin hybrid LAMMPS simulation of spherocylinder-like 
rods using the "lammps_multistate_rods" library.''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('config_file', help='path to the "lammps_multistate_rods" rods config file')
parser.add_argument('output_folder', help='name for the folder that will be created for output files')

parser.add_argument('-T', '--temp', default=1.0, type=float, help='the temperature of the system (e.g. for Langevin)')
parser.add_argument('-D', '--damp', default=0.1, type=float, help='viscous damping (for Langevin)')
parser.add_argument('-C', '--cell_size', default=10.0, type=float, help='size of an SC cell (i.e. room for one rod)')
parser.add_argument('-N', '--num_cells', default=5.0, type=float, help='the number of cells per dimension')

parser.add_argument('-S', '--sim_length', default=200000, type=int, help='the total number of MD steps to simulate')
parser.add_argument('-R', '--run_length', default=200, type=int, help='number of MD steps between MC moves')
parser.add_argument('-M', '--MC_moves', default=1.0, type=float, help='number of MC moves per rod between MD runs')
parser.add_argument('--seed', type=int, help='the seed for random number generators')

parser.add_argument('-o', '--output_freq', type=int, help='''configuration output frequency (in MD steps);
default behavior is before and after every batch of MC moves''')
parser.add_argument('-s', '--silent', action='store_true', help="doesn't print anything to stdout and doesn't make a lammps log file")

args = parser.parse_args()

#========================================================================================

import os
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

from lammps import PyLammps
import lammps_multistate_rods as rods

if args.seed is None:
    import time
    args.seed = int((time.time() % 1)*1000000)
    print "WARNING: no seed given explicitly; using:", args.seed

py_lmp = PyLammps()
if args.silent:
    py_lmp.log("none")
else:
    py_lmp.log(os.path.join(args.output_folder, 'lammps.log'))

max_rod_type = rods.init(py_lmp, args.config_file, args.output_folder)

# DEFINE SIMULATION BOX
py_lmp.boundary("p p p")
py_lmp.lattice("sc", 1./(args.cell_size**3))
box_size = float(args.num_cells)
py_lmp.region("box", "block", -box_size / 2, box_size / 2, -box_size / 2, box_size / 2, -box_size / 2, box_size / 2)
py_lmp.create_box(max_rod_type, "box")

# CREATE PARTICLES
#  - create here any other, non-rod particles (e.g. a membrane), before calling setup_rods_simulation
rods.setup_simulation(args.seed, args.temp, box = None)

# SET DYNAMICS
py_lmp.fix("thermostat", "all", "langevin", args.temp, args.temp, args.damp, args.seed)#, "zero yes")
rods.set_dynamics("nve")

# OUTPUT
py_lmp.dump("dump_cmd", "all", "custom", 1, os.path.join(args.output_folder, str(args.seed)+'.dump'), "mol type x y z")
if (args.output_freq != None):
    py_lmp.dump_modify("dump_cmd", "every", args.output_freq, "sort id", "pad 5")
else:
    py_lmp.variable("output_steps", "equal", "stagger({:d},1)".format(args.run_length))
    py_lmp.dump_modify("dump_cmd", "every v_output_steps", "first yes", "sort id", "pad 5")

if not args.silent:
    py_lmp.thermo_style("custom", "step atoms", "pe temp")
    py_lmp.thermo(args.run_length)

### SETUP COMPLETE ###

mc_moves_per_run = int(args.MC_moves * rods.nrods())

if mc_moves_per_run == 0:
    
    py_lmp.command('run {:d}'.format(args.sim_length))

else:
    
    for i in range(int(args.sim_length/args.run_length)-1):
        
        py_lmp.command('run {:d}'.format(args.run_length))
        
        success = rods.conformation_Monte_Carlo(mc_moves_per_run)
        
        if not args.silent:
            base_count = rods.state_count(0)
            beta_count = rods.state_count(1)
            print 'step {:d} / {:d} :  beta-to-base ratio = {:d}/{:d} = {:.5f} (accept rate = {:.5f})'.format(
                    (i+1)*args.run_length, args.sim_length, beta_count, base_count,
                        float(beta_count)/base_count, float(success)/mc_moves_per_run)
            
    py_lmp.command('run {:d}'.format(args.run_length))

