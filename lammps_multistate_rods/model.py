# encoding: utf-8
'''
This module holds just that same-name class, refer to its description.

Created on 17 Jul 2018

@author: Eugen Rožić
'''

import os
from math import cos, sin, pi
import numbers

vx = 'vx'
global_context = {'__builtins__': None, 'vx': vx} #for eval & exec (to not be able to do funny stuff)

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
        local_context = {} #for controlled and correct eval & exec
        # ROD PROPERTIES (available to set in the config file)
        local_context['rod_radius'] = 1.0 #default
        local_context['rod_length'] = None #default is 8*rod_radius (after rod_radius is (re)defined)
        local_context['rod_mass'] = 1.0 #default
        local_context['rod_states'] = None #example: ('soluble_state', 'beta_state')
        local_context['num_states'] = None #dependent on "rod_states"
        local_context['state_structures'] = None #dependent on "rod_states"
            #example of elements:
            # state_structures[0] = '1-1-1-1-1-1-2|3-3-3-3' # '1' is inert body type, '3' is inert side-patch type
            # state_structures[1] = '1-1-1-1-1-1-1|4-4-4-4' # '2' is active body type, '4' is active side-patch type
        local_context['patch_angles'] = []
        local_context['patch_bead_radii'] = []
        local_context['patch_bead_sep'] = []
        local_context['patch_bulge_out'] = 0.0 #default
        # INTERACTION PROPERTIES (available to set in the config file)
        local_context['int_types'] = None # interaction types (with parameters)
            #example:
            # int_types = {'patch':('cosine/squared', 1.75*rod_radius),
            #              'tip':('cosine/squared', 1.0*rod_radius, 'wca'),
            #              'vx':('lj/cut', 0.0)}
        local_context['eps'] = {} # interaction strengths between bead types
            #example of elements:
            # eps[(1,1)] = eps[(1,2)] = eps[(1,3)] = eps[(1,4)] = (5.0, 'vx')
            # eps[(2,3)] = eps[(3,3)] = eps[(3,4)] = (5.0, 'vx')
            # eps[(2,2)] = (3.25, 'tip') # soluble-soluble tip interaction
            # eps[(2,4)] = (6.5, 'patch') # soluble-beta interaction
            # eps[(4,4)] = (30.0, 'patch') # beta-beta interaction
        local_context['trans_penalty'] = {} # transition penalties between states
            #example of elements:
            # trans_penalty[(0,1)] = 15.0 # soluble-beta transition
        
        with open(config_file_path,'r') as config_file:
            command = ''
            for line in config_file:
                line = line.strip()
                try:
                    line = line[:line.index('#')]
                except ValueError: #no '#' in line
                    pass
                if line == '':
                    continue
                command += line
                if line.endswith(','):
                    continue
                parts = command.split('=')
                assign = parts[0].strip()
                expr = parts[1].strip()
                if assign == 'rod_states':
                    local_context['rod_states'] = eval(expr, global_context, local_context)
                    if not isinstance(local_context['rod_states'], (tuple, list)):
                        raise Exception('"rod_states" has to be either a tuple or a list!')
                    local_context['num_states'] = len(local_context['rod_states'])
                    local_context['state_structures'] = ['']*local_context['num_states']
                else: #allow whatever command, support variables to be defined etc.
                    exec(command, global_context, local_context)
                command = ''
        
        self.rod_radius = local_context['rod_radius']
        self.rod_length = local_context['rod_length']
        if self.rod_length is None:
            self.rod_length = 8.0*self.rod_radius
        self.rod_mass = local_context['rod_mass']
        self.rod_states = local_context['rod_states']
        self.num_states = local_context['num_states']
        self.state_structures = map(lambda y: map(lambda x: map(int, x.split('-')),
                                                  y.split('|')),
                                    local_context['state_structures'])
        self.body_beads = None #dependent on "state_structures"
        self.body_bead_types = None #dependent on "state_structures"
        self.body_bead_overlap = None
        self.num_patches = None #dependent on "state_structures"
        self.patch_angles = local_context['patch_angles']
        self.patch_bead_radii = local_context['patch_bead_radii']
        self.patch_beads = None #dependent on "state_structures"
        self.patch_bead_types = None #dependent on "state_structures"
        self.patch_bead_sep = local_context['patch_bead_sep']
        self.patch_bulge_out = local_context['patch_bulge_out']
        self.total_beads = None #dependent on "state_structures"
        self.all_bead_types = None #dependent on "state_structures"
        self.active_bead_types = None #dependent on "state_structures" & "eps"
        self.max_bead_type = None #dependent on "state_structures"
        self.int_types = local_context['int_types']
        self.global_cutoff = 3*self.rod_radius
        self.eps = local_context['eps']
        self.trans_penalty = local_context['trans_penalty']
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
            
            if self.body_beads == None:
                self.body_beads = len(state_struct[0])
            elif len(state_struct[0]) != self.body_beads:
                raise Exception('All states must have the same number of body beads!')
            
            self.body_bead_types.update(state_struct[0])
            
            if self.patch_beads == None:
                self.patch_beads = map(len, state_struct[1:])
            elif map(len, state_struct[1:]) != self.patch_beads:
                raise Exception('All states must have the same number of patch int sites!')
            
            if self.patch_bead_types == None:
                self.patch_bead_types = [set() for _ in range(1, len(state_struct))]
            for i in range(1, len(state_struct)):
                self.patch_bead_types[i-1].update(state_struct[i])
        
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
#         if len(self.all_bead_types) != len(all_types):
#             raise Exception("One bead type can appear only in the same patch or the body!")
        
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
            print 'WARNING: No "'+vx+'" interaction defined in the config file! Using "lj/cut" default...'
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
            output_path = os.path.join(model_output_dir, self.rod_states[state]+'.mol')
            if os.path.exists(output_path):
                print "WARNING: {:s} already exists, won't overwrite it...".format(output_path)
                continue
            with open(output_path, "w") as mol_file:
                
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
                for patch in self.state_structures[state]:
                    for bead_type in patch:
                        mol_file.write("{:2d} {:d}\n".format(n, bead_type))
                        n += 1
                
                mol_file.write("\nBonds\n\n")
                for i in range(1, self.total_beads):
                    mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(i, i, i+1))
                mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(self.total_beads, self.total_beads, 1))
    