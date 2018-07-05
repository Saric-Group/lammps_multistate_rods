# encoding: utf-8
'''
This module contains the description of a model for multi-state rods
as multi-bead molecules in LAMMPS and the tools necessary for using
it in simulations where the rods dynamically change their state.

In order to use the model properly one has to call the "init" method with
a fresh instance of LAMMPS, before any other LAMMPS commands are called.

The general description of a model is given in a configuration file that
needs to be passed to the "init" method.

Created on 22 Mar 2018

@author: Eugen Rožić
'''
import os
from math import exp, sqrt, pi
import random
from numpy import array as np_array

from lammps import PyLammps
from ctypes import c_double
import re

#########################################################################################
### PARAMETERS ##########################################################################

# rod properties
rod_states = None #example: ('base_state', 'beta_state')
num_states = None #dependent on "rod_states"
state_structures = None #dependent on "rod_states"
    #example of elements:
    # state_structures[0] = '1111112|333333' # '1' is inert body type, '3' is inert side-patch type
    # state_structures[1] = '1111111|444444' # '2' is active body type, '4' is active side-patch type
rod_radius = 1.0 #default
body_beads = None #dependent on "state_structures"
body_bead_types = None #dependent on "state_structures"
body_bead_overlap = 1.0*rod_radius #default
int_radius = 0.5*rod_radius #default
int_sites = None #dependent on "state_structures"
int_bead_types = None #dependent on "state_structures"
int_bead_overlap = 0.0 #default
int_bulge_out = 0.1*rod_radius #default
total_beads = None #dependent on "state_structures"
rod_mass = 1.0 #default

# interaction properties
int_type = None #example: ('morse', 2.5/rod_radius)
vol_exclusion = None #example: 10 (strength of bead/particle repulsion)
int_range = 1.5*rod_radius #default (separation between bead/particle boundaries at which interaction =0)
global_cutoff = 3.0*rod_radius #default
vx = 'vx' #constant (for labeling vol_exclusion interaction)
eps = {} # (interaction strengths between bead types)
    #example of elements:
    # eps[(1,1)] = eps[(1,2)] = eps[(1,3)] = eps[(1,4)] = vx
    # eps[(2,3)] = eps[(3,3)] = eps[(3,4)] = vx
    # eps[(2,2)] = 5.5 # base-base interaction
    # eps[(2,4)] = 5.75 # base-beta interaction
    # eps[(4,4)] = 6.0 # beta-beta interaction
trans_penalty = {} # (transition penalties between states)
    #example of elements:
    # trans_penalty[(0,1)] = 15.0 (base to beta state transition)


def set_dependent_params():
    '''
    Sets the global variables of the module that are used in the rest of the module
    but are derived from the essential information given by a user, specifically:
        - information contained in the string specification of the states (+ a consistency check)
        - overlap (delta) of interaction sites (calculated from the number and size of beads and interaction sites)
        - the anti-symmetric partners in the "trans_penalty" matrix (trans_penalty[(n,m)] = -trans_penalty[(m,n)])
    '''
    global body_bead_types, int_bead_types, body_beads, int_sites, total_beads
    body_bead_types = set()
    int_bead_types = set()
    body_beads = len(state_structures[0].split('|')[0])
    int_sites = len(state_structures[0].split('|')[1])
    for state_struct in state_structures: #check all have same "form"
        temp1, temp2 = state_struct.split('|')
        if len(temp1) != body_beads:
            raise Exception('All states must have the same number of body beads!')
        if len(temp2) != int_sites:
            raise Exception('All states must have the same number of interaction sites!')
        for body_bead_type in temp1:
            body_bead_types.add(int(body_bead_type))
        for int_bead_type in temp2:
            int_bead_types.add(int(int_bead_type))
    total_beads = body_beads + int_sites
    body_bead_types = list(body_bead_types)
    int_bead_types = list(int_bead_types)
    
    #global int_bead_overlap
    #int_bead_overlap = 2 - ((body_beads - 2)*(2 - body_bead_overlap)*rod_radius)/((int_sites - 1)*int_radius)
    
    antisym_completion = {}
    for (from_state, to_state), value in trans_penalty.iteritems():
        antisym_completion[(to_state, from_state)] = -value
    trans_penalty.update(antisym_completion)

#########################################################################################
### INITIALISATION ######################################################################

def set_model_params(config_file_path):
    '''
    Sets the values of this module's global variables which correspond
    to the parameters of the model... TODO
    '''
    with open(config_file_path,'r') as config_file:
        for line in config_file:
            line = line.strip()
            if (line.startswith('#') or line == ''):
                continue
            parts = line.split('=')
            assign = parts[0].strip()
            expr = parts[1].strip()
            if assign == 'rod_states':
                global rod_states, num_states, state_structures
                rod_states = eval(expr)
                num_states = len(rod_states)
                state_structures = ['']*num_states
            elif assign in ('rod_radius', 'body_bead_overlap', 'int_radius', 'int_bead_overlap', 'int_bulge_out',
                            'rod_mass', 'int_type', 'vol_exclusion', 'int_range', 'global_cutoff'):
                globals()[assign] = eval(expr)
            elif re.compile(r'state_structures\[\d+\]').match(assign) != None:
                exec line
            elif re.compile(r'eps\[\(\d+,\d+\)\]').match(assign) != None:
                exec line
            elif re.compile(r'trans_penalty\[\(\d+,\d+\)\]').match(assign) != None:
                exec line
            else:
                raise Exception('ERROR: Unknown config parameter encountered (' + line + ')')

    set_dependent_params()

def generate_model(model_output_dir):
    '''Generates the <model_output_dir>/<state>.mol files, where <state> represents
    one possible state of a rod and is a string element of the global
    "rod_states" list, which describe the various states of rods as
    different molecules in LAMMPS.
    The values of the parameters which are used to generate the model
    files can be set with "set_model_params". 
    '''
    if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
    for state in range(num_states):
        with open(os.path.join(model_output_dir, rod_states[state]+'.mol'), "w") as mol_file:
            mol_file.write("(AUTO-GENERATED file from lammps_multistate_rods.py, any changes will be OVERWRITTEN)\n\n")
            mol_file.write("{:d} atoms\n\n".format(total_beads))
            mol_file.write("Coords\n\n")
            for i in range(body_beads):
                x = 0.0 - ((body_beads - 2*i - 1) / 2.)*(2*rod_radius - body_bead_overlap)
                mol_file.write("{:2d} {:6.3f}  0.000  0.000\n".format((i+1), x))
            for i in range(int_sites):
                x = 0.0 - ((int_sites - 2*i - 1) / 2.)*(2*int_radius - int_bead_overlap)
                z = rod_radius - int_radius + int_bulge_out
                mol_file.write("{:2d} {:6.3f}  0.000 {:6.3f}\n".format((body_beads+i+1), x, z))
            mol_file.write("\nTypes\n\n")
            for i in range(body_beads):
                mol_file.write("{:2d} {:s}\n".format((i+1), state_structures[state][i]))
            for i in range(int_sites):
                mol_file.write("{:2d} {:s}\n".format((body_beads+i+1), state_structures[state][body_beads+i+1]))
    
def init(py_lammps, config_file_path, model_output_dir):
    '''
    This method needs to be called before any LAMMPS command.
    It generates files containing the description of the model using parameters given in the
    file at <config_file_path>. These files are generated as "<model_output_dir>/<state>.mol"
    LAMMPS molecule templates with <state> as their respective ID's.
    The method also sets the basic LAMMPS settings it relies upon to be set to exactly
    those values, specifically:
        - units lj
        - dimensions 3
        - atom_style molecular
    It also saves a reference to the passed PyLammps object to
    be used by other methods later. If another instance of LAMMPS wants
    to be used the model has to be "reloaded" (init + setup etc.)
    
    Returns : the maximum particle type number used in the config file
    '''
    
    set_model_params(config_file_path)
    generate_model(model_output_dir)
    
    global py_lmp
    py_lmp = py_lammps
    
    py_lmp.units("lj")
    py_lmp.dimension(3)
    
    py_lmp.atom_style("molecular")
    for state_name in rod_states:
        py_lmp.molecule(state_name, os.path.join(model_output_dir, state_name+'.mol'))
        
    return max([max(body_bead_types), max(int_bead_types)])

# LAMMPS group names

master_group_name = "_rods" # name of the group all rods belong to

def group_name(state_id):
    '''
    Name of the group of rods in the given state
    '''
    return rod_states[state_id] + master_group_name

# LAMMPS global variable names

def count_var(state_id):
    return rod_states[state_id] + "_count"


def setup_simulation(seed, temp, **kwargs):
    '''
    This method creates the rods, and supporting groups and variables, for the
    simulation and sets all the LAMMPS parameters related to them, e.g. masses,
    interactions etc.
    
    The method supports specifying different ways of creating the rods by passing
    one of the following optional parameters:
        box = None (DEFAULT)
        region = <region_ID>
        random = (N, <region_ID>)
        
    IMPORTANT: This method should be called after the creation of all other
    non-rod particles, since rods are expected by the model to be created last.
    '''
    #take into account existing particles and particle types...
    global offset, type_offset
    offset = py_lmp.lmp.get_natoms() #number of atoms before the creation of rods
    if offset > 0:
        py_lmp.variable("types", "atom", "type")
        py_lmp.compute("max_atom_type", "all", "reduce", "max v_types")
        type_offset = int(py_lmp.lmp.extract_compute("max_atom_type", 0, 0)) #max type before creation of rods
    else:
        type_offset = 0
        
    global sim_T
    sim_T = temp
    
    # set masses
    for bead_type in body_bead_types:
        py_lmp.mass(bead_type + type_offset, rod_mass/body_beads)
    for bead_type in int_bead_types:
        py_lmp.mass(bead_type + type_offset, rod_mass*10**-10) #interaction sites are massless, only body beads contribute to mass
    #TODO give mass to types that are not used... !!
    
    # set interaction
    py_lmp.pair_style(int_type[0], global_cutoff)
    py_lmp.pair_modify("shift yes")
    for bead_types, epsilon in eps.iteritems():
        sigma = 0
        for bead_type in bead_types:
            if bead_type in body_bead_types:
                sigma += rod_radius
            else:
                sigma += int_radius
        type_1 = bead_types[0] + type_offset
        type_2 = bead_types[1] + type_offset
        if int_type[0] == 'lj/cut':
            if epsilon == vx:
                py_lmp.pair_coeff(type_1, type_2, vol_exclusion*sim_T, sigma/pow(2,1./6), sigma)
            else:
                py_lmp.pair_coeff(type_1, type_2, epsilon*sim_T, sigma/pow(2,1./6), sigma + int_range)
        elif int_type[0] == 'nm/cut':
            if epsilon == vx:
                py_lmp.pair_coeff(type_1, type_2, vol_exclusion*sim_T, sigma, int_type[1], int_type[2], sigma)
            else:
                py_lmp.pair_coeff(type_1, type_2, epsilon*sim_T, sigma, int_type[1], int_type[2], sigma + int_range)
        elif int_type[0] == 'morse':
            if epsilon == vx:
                py_lmp.pair_coeff(type_1, type_2, vol_exclusion*sim_T, int_type[1], sigma, sigma)
            else:
                py_lmp.pair_coeff(type_1, type_2, epsilon*sim_T, int_type[1], sigma, sigma + int_range)
        elif int_type[0] == 'gauss/cut':
            if epsilon == vx:
                H = -vol_exclusion*sqrt(2*pi)*int_type[1]
                py_lmp.pair_coeff(type_1, type_2, H*sim_T, sigma, int_type[1], sigma)
            else:
                H = -epsilon*sqrt(2*pi)*int_type[1]
                py_lmp.pair_coeff(type_1, type_2, H*sim_T, sigma, int_type[1], sigma + int_range)
        else:
            raise Exception('Unknown/invalid int_type parameter: '+ str(int_type))
    
    # create rods
    if ("region" in kwargs.keys()):
        py_lmp.create_atoms(type_offset, "region", kwargs['region'], "mol", rod_states[0], seed)
    elif ("random" in kwargs.keys()):
        vals = kwargs['random']
        py_lmp.create_atoms(type_offset, "random", vals[0], seed, vals[1], "mol", rod_states[0], seed)
    else:
        py_lmp.create_atoms(type_offset, "box", "mol", rod_states[0], seed)
    
    # create and populate groups
    py_lmp.group(master_group_name, "id >", offset) # contains all rods, regardless of state
    py_lmp.group(group_name(0), "id >", offset) # at start all rods are in base_state
    for state in range(1, num_states):
        py_lmp.group(group_name(state), "empty")

    # create variables
    for state in range(num_states):
        py_lmp.variable(count_var(state), "equal", "count("+group_name(state)+")")

#########################################################################################
### SIMULATION TOOLS ####################################################################

def total_pe():
    '''
    Returns the total (non-normalised) potential energy of the system
    '''
    return py_lmp.lmp.extract_compute("thermo_pe", 0, 0)

def nrods():
    '''
    Returns the total number of rods in the simulation
    '''
    return int((py_lmp.lmp.get_natoms() - offset) / total_beads)
    #TODO use "extract_global(natoms)" maybe? (a permanent valid pointer)

def state_count(state_id):
    '''
    Returns the number of rods in the state given by ID.
    '''
    return int(py_lmp.lmp.extract_variable(count_var(state_id), "NULL", 0) / total_beads)

def get_random_rod():
    '''
    Returns a randomly picked rod(molecule) from LAMMPS.
    
    returns : a lammps_multistate_rods.Rod object
    
    This method also presumes that all rods are of equal length and that
    both atom and molecule ids are sequential (no holes and/or rearrangements
    in the numbering).
    '''
    mol_id = random.randint(1, nrods()) # random number in [1, nrods()]
    
    rod_start_index = offset + (mol_id - 1) * total_beads
    rod_atom_indices = range(rod_start_index, rod_start_index + total_beads)
    
    return Rod(mol_id, rod_atom_indices, py_lmp)


def set_dynamics(ensemble = "", **kwargs):
    '''
    Sets a "rigid/<ensemble>/small" integrator for all the rods (default is just "rigid/small")
    
    Any additional LAMMPS "keyword" options (e.g. langevin, temp, iso etc.) can be passed as
    named arguments in the following form:
    keyword = (value_1, value_2, ...)
    '''
    keyword_options = ""
    for key, values in kwargs.iteritems():
        keyword_options += key
        for value in values:
            keyword_options += " " + str(value)
        keyword_options += "\t"
        
    ensemble = ensemble.strip().lower()
    fix_name = "rigid/"+ensemble+"/small" if ensemble != "" else "rigid/small"
    
    py_lmp.fix("rod_dynamics", master_group_name, fix_name, "molecule", "mol", rod_states[0], keyword_options) 
    
    py_lmp.neigh_modify("exclude", "molecule/intra", master_group_name)
    
#     if (gcmc != None):
#         py_lmp.fix("rod_supply", base_group, "gcmc", gcmc[0], gcmc[1], 0, 0, seed, temp, gcmc[2], 0.0,
#                    "mol", rod_states[0], "rigid rod_dynamics", "region", gcmc[3], "group", master_group_name)
#         
#         py_lmp.compute_modify("thermo_temp", "dynamic/dof yes")
#         py_lmp.fix_modify("rod_dynamics", "dynamic/dof yes") #only necessary for rigid/nvt


def try_conformation_change(rod, U_before):
    '''
    Tries an MC conformation change on the given rod. The change
    is accepted with probability equal to:
        max{ exp(- delta_U - trans_penalty), 1}
    where:
        delta_U = U_after - U_before
    and the sign of trans_penalty is (-) if changing from base_state to
    beta_state and (+) if changing from beta_state to base_state.
    
    WARNING: this method leaves LAMMPS in an inconsistent state regarding
    thermodynamic variables (i.e. the state of atoms might not correspond
    with current "pe", because when a move is rejected the energy is not
    calculated again after restoring old rod state)
    
    returns : (1, U_after) or (0, U_before)
    '''
    old_state = rod.state
    new_state = (old_state + random.randrange(-1, 2, 2)) % num_states # cyclic, with certainty a try will be made
    rod.set_state(new_state)
    
    py_lmp.command('run 0 post no')
    U_after = total_pe()
    accept_prob = exp(- (U_after - U_before)/sim_T - trans_penalty[(old_state,new_state)])
    
    if (random.random() < accept_prob):
        py_lmp.group(group_name(new_state), "molecule", rod.id) # add to the group of current state
        py_lmp.group(group_name(old_state), "subtract", group_name(old_state), group_name(new_state)) # remove from the group of previous state
        return (1, U_after)
    else:
        rod.set_state(old_state) # revert change back
        return (0, U_before)


def conformation_Monte_Carlo(ntries):
    '''
    Tries to make "ntries" Monte Carlo conformation changes on randomly selected rods that are
    presumed to be equilibriated to "temp" temperature.
    
    returns : the number of accepted moves
    '''
    U_current = total_pe()
    success = 0
    for _ in range(ntries):
        rod = get_random_rod()
        (acpt, U_current) = try_conformation_change(rod, U_current)
        success += acpt
    
    return success
        
#########################################################################################
        
class Rod(object):
    '''
    Represents a single rod as a list of LAMMPS atom indices.
    '''
    
    def __init__(self, mol_id, atom_indices, py_lmp):
        if not isinstance(py_lmp, PyLammps):
            raise Exception("py_lmp has to be an instance of lammps.PyLammps!")
        self.py_lmp = py_lmp
        self.id = mol_id
        self.atom_indices = atom_indices
        self._determine_state()
        
    
    def _determine_state(self):
        self.state = None
        self.all_atom_types = self.py_lmp.lmp.gather_atoms("type", 0, 1)
        
        rod_body_beads = ''.join( str(self.all_atom_types[index] - type_offset) for index in self.atom_indices[:body_beads])
        rod_int_beads = ''.join( str(self.all_atom_types[index] - type_offset) for index in self.atom_indices[body_beads:])
        rod_bead_types = rod_body_beads + '|' + rod_int_beads
        for state in range(num_states):
            if (state_structures[state] == rod_bead_types):
                self.state = state
        if (self.state == None):
            raise Exception("Non-existing rod state encountered!")
    
    def set_state(self, new_state):
        '''
        Changes the state of the rod to the given one.
        This change is directly reflected in LAMMPS (without any removal
        or addition of particles - just change of type)!
        '''
        self.state = new_state
        
        rod_atom_types = [int(atom_type) + type_offset for atom_type in state_structures[self.state].replace('|','')]
    
        for i in range(total_beads):
            self.all_atom_types[self.atom_indices[i]] = rod_atom_types[i]
        
        self.py_lmp.lmp.scatter_atoms("type", 0, 1, self.all_atom_types)
    
    def _get_positions(self):
        #TODO
        raise Exception('Not implemented')
    
    def _get_velocities(self):
        #TODO
        raise Exception('Not implemented')
    
    def delete(self):
        '''
        Deletes the rod (by molecule ID) from LAMMPS.
        This method also saves ALL information about the rod atoms, which is VERY costly,
        so this method shouldn't be used if possible. This is necessary, however, in
        order to be able to properly create a new rod subsequently.
        '''
        self._get_positions()
        self._get_velocities()
        self.all_atom_images = self.py_lmp.lmp.gather_atoms("image", 0, 1)
        
        self.py_lmp.group("temp_group", "molecule", self.id)
        self.py_lmp.delete_atoms("group", "temp_group")

    def create(self):
        '''
        Creates a rod in LAMMPS described by this object (including atom ID's).
        '''
        n = total_beads
        ids = [(index+1) for index in self.atom_indices]
        types = [int(atom_type) + type_offset for atom_type in state_structures[self.state].replace('|','')]
        xs = ((c_double * 3) * n)()
        vs = ((c_double * 3) * n)()
        for i in range(n):
            xs[i][:] = self.atom_positions[i]
            vs[i][:] = self.atom_velocities[i]
        imgs = [self.all_atom_images[index] for index in self.atom_indices]
        
        self.py_lmp.lmp.create_atoms(n, ids, types, xs, vs, imgs, shrinkexceed=False)

        self.py_lmp.set("atom", "{}*{}".format(ids[0],ids[-1]), "mol", self.id) # sets molecule id for new atoms
    
    def location(self):
        '''
        Returns the location of the central body bead (assuming odd number of them).
        '''
        try:
            self.atom_positions
        except NameError:
            self._get_positions()
            
        central_atom = int((body_beads - 1) / 2)
        
        return self.atom_positions[central_atom]
    
    def orientation(self):
        '''
        Returns a numpy vector pointing from the location of the first body bead
        to the location of the last body bead.
        '''
        try:
            self.atom_positions
        except NameError:
            self._get_positions()
        
        first_atom = np_array(self.atom_positions[0])
        last_atom = np_array(self.atom_positions[body_beads-1])
        return (last_atom - first_atom)
        
