# encoding: utf-8
'''
This module holds just that same-name class, refer to its description.

Created on 17 Jul 2018

@author: Eugen Rožić
'''

from ctypes import c_int

class Rod(object):
    '''
    Represents a single rod as a list of LAMMPS atom indices in a defined
    lammps_multistate_rods.simulation.Simulation.
    '''
    
    def __init__(self, simulation, rod_id, bead_ids, state = None):
        self._sim = simulation
        self._model = simulation.model
        self.id = rod_id
        self.bead_ids = (self._model.total_beads*c_int)(*bead_ids)
        self.state = state
        if state == None:
            self._determine_state()
    
    def _determine_state(self):
        curr_state_types = list(self._sim.py_lmp.lmp.gather_atoms_subset(
            "type", 0, 1, self._model.total_beads, self.bead_ids))
        for state in range(self._model.num_states):
            if (list(self._sim._state_types[state]) == curr_state_types):
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
        
        new_bead_types = self._sim._state_types[new_state]
        
        self._sim.py_lmp.lmp.scatter_atoms_subset("type", 0, 1, self._model.total_beads,
                                                  self.bead_ids, new_bead_types)
