# encoding: utf-8
'''
This module holds the same-name class, refer to its description.

Created on 17 Jul 2018

@author: Eugen Rožić
'''

import os
from math import cos, sin, pi
import numbers

vx = 'vx'
_globcontext = {'__builtins__': None, 'vx': vx} #for eval & exec (to not be able to do funny stuff)

class Params(object):
    '''
    A class only to hold parameters as object variables, for easier/nicer access.
    '''
    def __init__(self, param_dict=None):
        if param_dict != None:
            self.__dict__.update(param_dict)

class Rod_model(object):
    '''
    This class describes a model for a lammps_multistate_rods simulation, i.e. it holds all
    the information about the parameters of the rods and the simulation that are parsed from
    a config file.
    This class is meant to be instantiated and used by instances of the Simulation and Rods
    classes of this library. An instance should be available to the user through an instance
    of the Simulation class.
    '''

    def __init__(self, config_file_path):
        '''
        Parses the config file and sets the parameters given within, and those derived
        from them, as object attributes.
        
        The derived variables/attributes are:
         - num_patches: the number of patches (NOT including the body)
         - num_beads: the number of beads in each patch (including the body)
         - total_beads: the total number of beads in a rod
         - state_bead_types: the set of types of beads by state
         - body_bead_types: the set of types of body beads (in all states)
         - patch_bead_types: the set of types of beads by patch (NOT including the body)
         - active_bead_types: the set of types of beads that have a non-vx interaction
         - all_bead_types: the set of all types of beads used in the .cfg file
         - body_bead_overlap: self-explanatory; calculated from rod_length
         - bead_radii: a dictionary of bead radii by bead type
         - transitions: a list by state_ID of lists of (state_ID, penalty) pairs for all allowed transitions
        '''
        cfg_params = Params() #for controlled and correct eval & exec
        # ROD PROPERTIES (available to set in the config file)
        cfg_params.rod_radius = 1.0
        cfg_params.rod_length = None #default is 8*rod_radius (after rod_radius is (re)defined)
        cfg_params.rod_mass = 1.0
        cfg_params.rod_states = None
        cfg_params.num_states = None #dependent on "rod_states"
        cfg_params.state_structures = None
        cfg_params.patch_angles = [] #default for 0 patches
        cfg_params.patch_bead_radii = [] #default for 0 patches
        cfg_params.patch_bead_sep = [] #default for 0 patches
        cfg_params.patch_bulge_out = 0.0 #default (for all patches)
        # INTERACTION PROPERTIES (available to set in the config file)
        cfg_params.int_types = None
        cfg_params.eps = {}
        cfg_params.trans_penalty = {}
        
        with open(config_file_path,'r') as config_file:
            command = ''
            for line in config_file:
                try:
                    line = line[:line.index('#')]
                except ValueError: #no '#' in line
                    pass
                line = line.strip()
                if line == '':
                    continue
                command += line
                if line.endswith(','):
                    continue
                try:
                    parts = command.split('=')
                    assign = parts[0].strip()
                    expr = parts[1].strip()
                    if assign == 'rod_states':
                        cfg_params.rod_states = eval(expr, _globcontext, vars(cfg_params))
                        if not isinstance(cfg_params.rod_states, (tuple, list)):
                            raise Exception('"rod_states" has to be either a tuple or a list!')
                        cfg_params.num_states = len(cfg_params.rod_states)
                        cfg_params.state_structures = ['']*cfg_params.num_states
                    else: #allow whatever command, support variables to be defined etc.
                        exec command in _globcontext, vars(cfg_params)
                except:
                    raise Exception('Something is wrong with the config file, in command: "'+
                                    command+'"')
                command = ''
        
        self.rod_radius = cfg_params.rod_radius
        self.rod_length = cfg_params.rod_length
        if self.rod_length is None:
            self.rod_length = 8.0*self.rod_radius
        self.rod_mass = cfg_params.rod_mass
        self.rod_states = cfg_params.rod_states
        self.num_states = cfg_params.num_states
        try:
            self.state_structures = map(lambda y: map(lambda x: map(int, x.split('-')),
                                                      y.split('|')),
                                        cfg_params.state_structures)
        except:
            raise Exception('The "state_structures" have to contain only integers \
                             separated by "|" or "-".')
        
        self.num_patches = None #dependent on "state_structures"
        self.num_beads = None #dependent on "state_structures"
        self.total_beads = None #dependent on "state_structures"
        self.state_bead_types = None #dependent on "state_structures"
        self.body_bead_types = None #dependent on "state_structures"
        self.patch_bead_types = None #dependent on "state_structures"
        self.active_bead_types = None #dependent on "state_structures" & "eps"
        self.all_bead_types = None #dependent on "state_structures"
        self.body_bead_overlap = None #dependent on "state_structures" &  "rod_length"
        self.patch_angles = cfg_params.patch_angles
        self.patch_bead_radii = cfg_params.patch_bead_radii
        self.bead_radii = None #dependent on "state_structures"
        self.patch_bead_sep = cfg_params.patch_bead_sep
        self.patch_bulge_out = cfg_params.patch_bulge_out
        self.int_types = cfg_params.int_types
        self.global_cutoff = 3*self.rod_radius #default value
        self.eps = cfg_params.eps
        self.trans_penalty = cfg_params.trans_penalty
        self.transitions = None #dependent on "trans_penalty";
        
        self._set_dependent_params()
    
    def _set_dependent_params(self):
        '''
        Sets the global variables of the instance that are used in the rest of the module
        but are derived from the essential information given by a user, specifically:
            - information contained in the string specification of the states (+ a consistency check)
            - the anti-symmetric partners in the "trans_penalty" matrix (trans_penalty[(n,m)] = -trans_penalty[(m,n)])
            - the "transitions" list from the "trans_penalty" dictionary
        '''
        self.state_bead_types = [None]*self.num_states
        self.bead_radii = {}
        for n in range(self.num_states):
            state_struct = self.state_structures[n]
            #check all have the same "form"            
            if self.num_beads == None:
                self.num_beads = map(len, state_struct)
                self.num_patches = len(self.num_beads) - 1
            elif map(len, state_struct) != self.num_beads:
                raise Exception('All states must have the same number of patches and beads in each patch!')
            
            if self.patch_bead_types == None:
                self.body_bead_types = set()
                self.patch_bead_types = [set() for _ in range(self.num_patches)]
            self.body_bead_types.update(state_struct[0])
            self.state_bead_types[n] = set(state_struct[0])
            for body_bead in state_struct[0]:
                temp = self.bead_radii.get(body_bead)
                if temp is None:
                    self.bead_radii[body_bead] = self.rod_radius
                elif temp != self.rod_radius:
                    raise Exception('Beads of the same type have to have the same radius!')
            for i in range(self.num_patches):
                self.patch_bead_types[i].update(state_struct[i+1])
                self.state_bead_types[n].update(state_struct[i+1])
                for patch_bead in state_struct[i+1]:
                    temp = self.bead_radii.get(patch_bead)
                    if temp is None:
                        self.bead_radii[patch_bead] = self.patch_bead_radii[i]
                    elif temp != self.patch_bead_radii[i]:
                        raise Exception('Beads of the same type have to have the same radius!') 
        
        self.total_beads = sum(self.num_beads)
        if self.total_beads == 1:
            print "WARNING: The rods contain only one bead in total - no bonds will be defined and the rigid fix won't work!"
            
        if self.num_beads[0] > 1:
            self.body_bead_overlap = ((2*self.rod_radius*self.num_beads[0] - self.rod_length) /
                                      (self.num_beads[0] - 1))
        else:
            self.body_bead_overlap = 0
        
        if len(self.patch_angles) != self.num_patches:
            raise Exception("The length of patch_angles doesn't match the number of defined patches!")
        if len(self.patch_bead_radii) != self.num_patches:
            raise Exception("The length of patch_bead_radii doesn't match the number of defined patches!")
        if len(self.patch_bead_sep) != self.num_patches:
            raise Exception("The length of patch_bead_sep doesn't match the number of defined patches!")
        if isinstance(self.patch_bulge_out, numbers.Number):
            self.patch_bulge_out = tuple([self.patch_bulge_out]*self.num_patches)
        elif len(self.patch_bulge_out) != self.num_patches:
            raise Exception("The length of patch_bulge_out doesn't match the number of defined patches!")
        
        all_types = [bead_type
                     for state_types in self.state_bead_types
                     for bead_type in state_types]
        self.all_bead_types = sorted(set(all_types))
        
        self.active_bead_types = set()
        eps_temp = self.eps; self.eps = {}
        # rectify "eps"'s keys (type_1 < type_2)
        for bead_types, eps_val in eps_temp.iteritems():
            if eps_val[1] != vx:
                self.active_bead_types.update(bead_types)
            if bead_types[0] <= bead_types[1]:
                self.eps[bead_types] = eps_val
            else:
                self.eps[(bead_types[1], bead_types[0])] = eps_val
        self.active_bead_types = list(self.active_bead_types)
        
        try:
            self.int_types[vx]
        except KeyError:
            print 'WARNING: No "'+vx+'" interaction defined in the config file! Using "lj/cut" default...'
            self.int_types[vx] = ('lj/cut', 0.0)
    
        antisym_completion = {}
        self.transitions = [[] for _ in range(self.num_states)]
        for (from_state, to_state), value in self.trans_penalty.iteritems():
            self.transitions[from_state].append((to_state, value))
            if self.trans_penalty.get((to_state, from_state)) == None:
                antisym_completion[(to_state, from_state)] = -value
                self.transitions[to_state].append((from_state, -value))
        self.trans_penalty.update(antisym_completion)

    def generate_mol_files(self, model_output_dir):
        '''
        Generates the <model_output_dir>/<state>.mol files, where <state> represents
        one possible state of a rod (an element of self.rod_states).
        These are important to define rods as molecules in LAMMPS.
        '''
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        for state in range(self.num_states):
            output_path = os.path.join(model_output_dir, self.rod_states[state]+'.mol')
            if os.path.exists(output_path):
                print "WARNING: {:s} already exists, won't overwrite it...".format(output_path)
                continue
            with open(output_path, "w") as mol_file:
                
                mol_file.write("(AUTO-GENERATED file by the lammps_multistate_rods library, any changes will be OVERWRITTEN)\n\n")
                
                mol_file.write("{:d} atoms\n\n".format(self.total_beads))
                if self.total_beads > 1:
                    mol_file.write("{:d} bonds\n\n".format(self.total_beads))
                
                mol_file.write("Coords\n\n")
                n = 1
                for i in range(self.num_beads[0]):
                    x = 0.0 - (((self.num_beads[0] - 2*i - 1) / 2.) *
                               (2*self.rod_radius - self.body_bead_overlap))
                    mol_file.write("{:2d} {:6.3f}  0.000  0.000\n".format(n, x))
                    n += 1
                for k in range(self.num_patches):
                    for i in range(self.num_beads[k+1]):
                        x = 0.0 - (((self.num_beads[k+1] - 2*i - 1) / 2.) *
                                   (2*self.patch_bead_radii[k] + self.patch_bead_sep[k]))
                        d = self.rod_radius - self.patch_bead_radii[k] + self.patch_bulge_out[k]
                        y = -sin(self.patch_angles[k]*2*pi/360)*d
                        z = cos(self.patch_angles[k]*2*pi/360)*d
                        mol_file.write("{:2d} {:6.3f} {:6.3f} {:6.3f}\n".format(n, x, y, z))
                        n += 1
                
                mol_file.write("\nTypes\n\n")
                n = 1
                for patch in self.state_structures[state]:
                    for bead_type in patch:
                        mol_file.write("{:2d} {:d}\n".format(n, bead_type))
                        n += 1
                
                if self.total_beads > 1:
                    mol_file.write("\nBonds\n\n")
                    for i in range(1, self.total_beads):
                        mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(i, i, i+1))
                    mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(self.total_beads, self.total_beads, 1))
    