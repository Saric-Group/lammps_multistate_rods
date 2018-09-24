# encoding: utf-8
'''
TODO

Created on 17 Jul 2018

@author: Eugen Rožić
'''

from ctypes import c_double
from numpy import array as np_array

class Rod(object):
    '''
    Represents a single rod as a list of LAMMPS atom indices in a defined
    lammps_multistate_rods.simulation.Simulation.
    '''
    
    def __init__(self, simulation, mol_id, atom_indices, state = None):
        self._sim = simulation
        self._model = simulation.model
        self.id = mol_id
        self.atom_indices = atom_indices
        self.state = state
        if state == None:
            self._determine_state()
    
    def _determine_state(self):
        curr_state_types = [self._sim._all_atom_types[index] for index in self.atom_indices]
        for state in range(self._model.num_states):
            if (self._sim._state_types[state] == curr_state_types):
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
        
        new_atom_types = self._sim._state_types[new_state]
        
        for index, new_type in zip(self.atom_indices, new_atom_types):
            self._sim._all_atom_types[index] = new_type
        
        self._sim.py_lmp.lmp.scatter_atoms("type", 0, 1, self._sim._all_atom_types)
        #self._sim._reset_active_beads_group()
        # it would be consistent to have this here, but very inefficient...
        # ...instead we can do it only if the new state is accepted (in _sim.try_state_change)
    
    def _get_positions(self):
        #TODO
        raise Exception('Not implemented')
    
    def _get_velocities(self):
        #TODO
        raise Exception('Not implemented')
    
    def _get_imgs(self):
        all_atom_images = self.py_lmp.lmp.gather_atoms("image", 0, 1)
        self.atom_imgs = [all_atom_images[index] for index in self.atom_indices]
    
    def delete(self):
        '''
        Deletes the rod (by molecule ID) from LAMMPS.
        This method also saves ALL information about the rod atoms, which is VERY costly,
        so this method shouldn't be used if possible. This is necessary, however, in
        order to be able to properly create a new rod subsequently.
        '''
        self._get_positions()
        self._get_velocities()
        self._get_imgs()
        
        #TODO - would have to update lists of rod elements in Simulation etc.
        raise Exception('Not implemented')

    def create(self):
        '''
        Creates a rod in LAMMPS described by this object (including atom ID's).
        '''
        n = self._model.total_beads
        ids = [(index+1) for index in self.atom_indices]
        types = self._sim._state_types[self.state]
        xs = ((c_double * 3) * n)()
        vs = ((c_double * 3) * n)()
        for i in range(n):
            xs[i][:] = self.atom_positions[i]
            vs[i][:] = self.atom_velocities[i]
        imgs = self.atom_imgs
        
        #self.py_lmp.lmp.create_atoms(n, ids, types, xs, vs, imgs, shrinkexceed=False)
        #self.py_lmp.set("atom", "{}*{}".format(ids[0],ids[-1]), "mol", self.id) # sets molecule id for new atoms
        #TODO - would have to update lists of rod elements in Simulation etc.
        raise Exception('Not implemented')
    
    def location(self):
        '''
        Returns the location of the central body bead (assuming odd number of them).
        '''
        try:
            self.atom_positions
        except NameError:
            self._get_positions()
            
        central_atom = int((self._model.body_beads - 1) / 2)
        
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
        last_atom = np_array(self.atom_positions[self._model.body_beads-1])
        return (last_atom - first_atom)
        
