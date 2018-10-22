# encoding: utf-8
'''
TODO

Created on 17 Jul 2018

@author: Eugen Rožić
'''

import os
from math import cos, sin, pi
import numbers

vx = 'vx'

class Model(object):
    '''
    This class describes a model for a lammps_multistate_rods simulation, i.e. it holds all
    the information about the parameters of the rods and the simulation that are parsed from
    a config file.
    This class is meant to be instantiated and used by instances of the Simulation and Rods
    classes of this library. An instance should be available to the user through an instance
    of the simulation class.
    '''

    def __init__(self, config_file_path):
        # ROD PROPERTIES (available to set in the config file)
        rod_radius = 1.0 #default
        rod_length = 8.0*rod_radius #default
        rod_mass = 1.0 #default
        rod_states = None #example: ('soluble_state', 'beta_state')
        num_states = None #dependent on "rod_states"
        state_structures = None #dependent on "rod_states"
            #example of elements:
            # state_structures[0] = '111111112|3333' # '1' is inert body type, '3' is inert side-patch type
            # state_structures[1] = '111111111|4444' # '2' is active body type, '4' is active side-patch type
        patch_angles = (0.0,) #default
        patch_bead_radii = None
        patch_bead_sep = None
        patch_bulge_out = 0.0 #default
        # INTERACTION PROPERTIES (available to set in the config file)
        int_types = None # interaction types (with parameters)
            #example:
            # int_types = {'patch':('cosine/squared', 1.75*rod_radius),
            #              'tip':('cosine/squared', 1.0*rod_radius, 'wca'),
            #              'vx':('lj/cut', 0.0)}
        eps = {} # interaction strengths between bead types
            #example of elements:
            # eps[(1,1)] = eps[(1,2)] = eps[(1,3)] = eps[(1,4)] = (5.0, 'vx')
            # eps[(2,3)] = eps[(3,3)] = eps[(3,4)] = (5.0, 'vx')
            # eps[(2,2)] = (3.25, 'tip') # soluble-soluble tip interaction
            # eps[(2,4)] = (6.5, 'patch') # soluble-beta interaction
            # eps[(4,4)] = (30.0, 'patch') # beta-beta interaction
        trans_penalty = {} # transition penalties between states
            #example of elements:
            # trans_penalty[(0,1)] = 15.0 # soluble-beta transition
            
        with open(config_file_path,'r') as config_file:
            command = ''
            for line in config_file:
                line = line.strip()
                if line.startswith('#') or line == '':
                    if command == '':
                        continue
                    else:
                        raise Exception('ERROR: comment or empty line in the middle of a multi-line command!')
                command += line
                if line.endswith(','):
                    continue
                parts = command.split('=')
                assign = parts[0].strip()
                expr = parts[1].strip()
                if assign == 'rod_states':
                    rod_states = eval(expr)
                    num_states = len(rod_states)
                    state_structures = ['']*num_states
#                 elif assign in ('rod_radius', 'rod_length', 'rod_mass',
#                                 'patch_angles', 'patch_bead_radii', 'patch_bead_sep',
#                                 'patch_bulge_out', 'int_types'):
#                     exec(command)
#                 elif re.compile(r'state_structures\[\d+\]').match(assign) != None:
#                     exec(command)
#                 elif re.compile(r'eps\[\(\d+,\d+\)\]').match(assign) != None:
#                     exec(command)
#                 elif re.compile(r'trans_penalty\[\(\d+,\d+\)\]').match(assign) != None:
#                     exec(command)
#                 else:
#                     raise Exception('ERROR: Unknown config parameter encountered (' + line + ')')
                else: #allow whatever command, support variables to be defined etc.
                    exec(command)
                command = ''
        
        self.rod_radius = rod_radius
        self.rod_length = rod_length
        self.rod_mass = rod_mass
        self.rod_states = rod_states
        self.num_states = num_states
        self.state_structures = state_structures
        self.body_beads = None #dependent on "state_structures"
        self.body_bead_types = None #dependent on "state_structures"
        self.body_bead_overlap = None
        self.num_patches = None #dependent on "state_structures"
        self.patch_angles = patch_angles
        self.patch_bead_radii = patch_bead_radii
        self.patch_beads = None #dependent on "state_structures"
        self.patch_bead_types = None #dependent on "state_structures"
        self.patch_bead_sep = patch_bead_sep
        self.patch_bulge_out = patch_bulge_out
        self.total_beads = None #dependent on "state_structures"
        self.all_bead_types = None #dependent on "state_structures"
        self.active_bead_types = None #dependent on "state_structures" & "eps"
        self.max_bead_type = None #dependent on "state_structures"
        self.int_types = int_types
        self.global_cutoff = 3*rod_radius
        self.eps = eps
        self.trans_penalty = trans_penalty
        self.transitions = None #dependent on "trans_penalty";
            # a list by state_ID of lists of (state_ID, penalty) pairs for all allowed transitions
        
        self._set_dependent_params()
    
    def _set_dependent_params(self):
        '''
        Sets the global variables of the instance that are used in the rest of the module
        but are derived from the essential information given by a user, specifically:
            - information contained in the string specification of the states (+ a consistency check)
            - the anti-symmetric partners in the "trans_penalty" matrix (trans_penalty[(n,m)] = -trans_penalty[(m,n)])
            - the "transitions" list from the "trans_penalty" dictionary
        '''
        self.body_bead_types = set()
        for state_struct in self.state_structures: #check all have the same "form"
            parts = state_struct.split('|')
            
            if self.body_beads == None:
                self.body_beads = len(parts[0])
            elif len(parts[0]) != self.body_beads:
                raise Exception('All states must have the same number of body beads!')
            
            self.body_bead_types.update(map(int, parts[0]))
            
            if self.patch_beads == None:
                self.patch_beads = map(len, parts[1:])
            elif map(len, parts[1:]) != self.patch_beads:
                raise Exception('All states must have the same number of patch int sites!')
            
            if self.patch_bead_types == None:
                self.patch_bead_types = [set() for _ in range(1, len(parts))]
            for i in range(1, len(parts)):
                self.patch_bead_types[i-1].update(map(int, parts[i]))
        
        self.total_beads = self.body_beads + sum(self.patch_beads)
        self.num_patches = len(self.patch_beads) 
        
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
        
        all_types = list(self.body_bead_types)
        for i_patch_types in self.patch_bead_types:
            all_types.extend(i_patch_types)
        self.all_bead_types = sorted(set(all_types))
        if len(self.all_bead_types) != len(all_types):
            raise Exception("One bead type can appear only in the same patch or the body!")
        
        self.max_bead_type = max(self.all_bead_types)
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
#         # fill all missing interactions with volume-exclusion
#         i = 0
#         while i < len(self.all_bead_types):
#             type_1 = self.all_bead_types[i]
#             j = i
#             while j < len(self.all_bead_types):
#                 type_2 = self.all_bead_types[j]
#                 try:
#                     self.eps[(type_1, type_2)]
#                 except KeyError:
#                     self.eps[(type_1, type_2)] = (self.vx_strength, vx)
#                 j += 1
#             i += 1
        
        self.body_bead_overlap = (2*self.rod_radius*self.body_beads - self.rod_length) / (self.body_beads - 1)
        
        try:
            self.int_types[vx]
        except KeyError:
            self.int_types[vx] = ('lj/cut', 0.0)
    
        antisym_completion = {}
        self.transitions = [[] for _ in range(self.num_states)]
        for (from_state, to_state), value in self.trans_penalty.iteritems():
            antisym_completion[(to_state, from_state)] = -value
            self.transitions[from_state].append((to_state, value))
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
            with open(os.path.join(model_output_dir, self.rod_states[state]+'.mol'), "w") as mol_file:
                
                mol_file.write("(AUTO-GENERATED file by the lammps_multistate_rods library, any changes will be OVERWRITTEN)\n\n")
                
                mol_file.write("{:d} atoms\n\n".format(self.total_beads))
                mol_file.write("{:d} bonds\n\n".format(self.total_beads))
                
                mol_file.write("Coords\n\n")
                n = 1
                for i in range(self.body_beads):
                    x = 0.0 - ((self.body_beads - 2*i - 1) / 2.)*(2*self.rod_radius - self.body_bead_overlap)
                    mol_file.write("{:2d} {:6.3f}  0.000  0.000\n".format(n, x))
                    n += 1
                for k in range(len(self.patch_beads)):
                    for i in range(self.patch_beads[k]):
                        x = 0.0 - ((self.patch_beads[k] - 2*i - 1) / 2.)*(2*self.patch_bead_radii[k] + self.patch_bead_sep[k])
                        d = self.rod_radius - self.patch_bead_radii[k] + self.patch_bulge_out[k]
                        y = -sin(self.patch_angles[k]*2*pi/360)*d
                        z = cos(self.patch_angles[k]*2*pi/360)*d
                        mol_file.write("{:2d} {:6.3f} {:6.3f} {:6.3f}\n".format(n, x, y, z))
                        n += 1
                
                mol_file.write("\nTypes\n\n")
                n = 1
                for bead_type in self.state_structures[state].replace('|',''):
                    mol_file.write("{:2d} {:s}\n".format(n, bead_type))
                    n += 1
                
                mol_file.write("\nBonds\n\n")
                for i in range(1, self.total_beads):
                    mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(i, i, i+1))
                mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(self.total_beads, self.total_beads, 1))
    