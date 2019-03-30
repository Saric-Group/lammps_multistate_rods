# encoding: utf-8
'''
This module provides methods to analyse data from a LAMMPS dump file of a simulation
of rods from the lammps_multistate_rods library in terms of rod clusters.

Created on 16 May 2018

@author: Eugen Rožić
'''

from lammps_multistate_rods import Simulation

def get_cluster_data(raw_data, every, model, type_offset, compute_ID=None):
    '''
    Extracts cluster data from the raw dump file data (expected to be sorted by particle ID).
    
    raw_data : output of the "parse_dump_file" method
    
    every : every which snapshot to analyse (e.g. 10 means 9 will be skipped after each analysed)
    
    model : the lammps_multistate_rods.Model class instance which was used to generate the data
    
    type_offset : the type offset for the rod model in the simulation that generated the dump file
    
    compute_ID: the ID of the LAMMPS cluster compute (default is "Simulation.cluster_compute")
    
    returns : a triplet of a list of timesteps, a list of box dimensions and a corresponding list of
    snapshot_data, where "snapshot_data" is a dictionary by cluster ID's whose values are lists of
    (rod/mol ID, rod state ID) pairs. If rod state ID is "None" the molecule is not a rod from the
    given model.
    
    NOTE: cluster ID's are set to the lowest rod/mol ID of each cluster, not the ones that are in
    the dump file
    '''
    states_types = [ [ elem + type_offset 
                        for patch in state_struct for elem in patch]
                            for state_struct in model.state_structures]
    
    def state_types_to_id(state_types):
        '''
        Returns rod state id from its structure (list of atom types), or None
        '''
        for i in range(model.num_states):
            if state_types == states_types[i]:
                return i
        return None
    
    if compute_ID == None:
        compute_ID = Simulation.cluster_compute
    
    count = 0
    box_sizes = []
    timesteps = []
    cluster_data = []
    for timestep, box_bounds, data_structure, data in raw_data:
        count += 1
        if (count-1) % every != 0:
            continue
        timesteps.append(timestep)
        box_sizes.append(map(lambda x: x[1]-x[0], box_bounds))
        snapshot_data = {}
        current_mol_id = None
        current_cluster_id = None
        current_rod = []
        for line_vars in data:
            mol_id = int(line_vars['mol'])
            if mol_id == 0:
                continue #just skip non-molecule particles...
            cluster_id = int(line_vars['c_'+compute_ID])
            if current_mol_id is None:
                current_mol_id = mol_id
            elif mol_id != current_mol_id:
                if current_cluster_id > 0:
                    current_rod_state = state_types_to_id(current_rod)
                    try:
                        snapshot_data[current_cluster_id].append((current_mol_id, current_rod_state))
                    except KeyError:
                        snapshot_data[current_cluster_id] = [(current_mol_id, current_rod_state)]
                current_mol_id = mol_id
                current_cluster_id = 0
                current_rod = []
            current_rod.append(int(line_vars['type']))
            if cluster_id > current_cluster_id:
                current_cluster_id = cluster_id    
        current_rod_state = state_types_to_id(current_rod)
        try:
            snapshot_data[current_cluster_id].append((current_mol_id, current_rod_state))
        except KeyError:
            snapshot_data[current_cluster_id] = [(current_mol_id, current_rod_state)]
                
        #switch keys to correspond to lowest mol_id in each cluster
        new_snapshot_data = {}
        for value in snapshot_data.values():
            new_snapshot_data[value[0][0]] = value
        cluster_data.append(new_snapshot_data)
    
    return timesteps, box_sizes, cluster_data

def write_cluster_data(timesteps, box_sizes, cluster_data, output_path):
    '''
    Used to write the data returned by "get_cluster_data" to a file, although the
    "cluster_data" can be a corresponding length list of any kind of objects whose
    string representation can be eval'd back to the object (e.g. dict, list, tuple, ...).
    '''
    with open(output_path, 'w') as out_file:
        for timestep, box_size, snapshot_data in zip(timesteps, box_sizes, cluster_data):
            out_file.write('{:^10d} | {:s} | {:s}\n'.format(
                timestep, str(box_size), str(snapshot_data)))

def read_cluster_data(input_path):
    '''
    Reads what was outputed with "write_cluster_data".
    
    returns : a triplet of a list of timesteps, a list of box dimensions and a corresponding
    list of objects (e.g. "snapshot_data", like the one returned by "get_cluster_data")
    '''
    timesteps = []
    box_sizes = []
    cluster_data = []
    
    with open(input_path, 'r') as in_file:
        for line in in_file:
            timestep, box_size, snapshot_data = line.split(' | ')
            timesteps.append(int(timestep))
            box_sizes.append(eval(box_size))
            cluster_data.append(eval(snapshot_data))
            
    return timesteps, box_sizes, cluster_data
        
#========================================================================================
# some possible ways of analysis of the raw_data extracted from a dump file...
#========================================================================================

def composition_by_states(cluster_data):
    '''
    cluster_data : a list of "snapshot_data", dictionaries by cluster ID whose values are lists of
    (rod/mol ID, rod state ID) pairs
    
    return : a list of dictionaries by cluster ID whose values are dictionaries by state ID of
    the numbers of rods in the corresponding state in the cluster.
    '''
    ret = [None]*len(cluster_data)
    i = 0
    for snapshot_data in cluster_data:
        ret_data = {}
        for cluster_ID, cluster in snapshot_data.iteritems():
            cluster_composition = {}
            for elem in cluster:
                try:
                    cluster_composition[elem[1]] += 1
                except KeyError:
                    cluster_composition[elem[1]] = 1
            ret_data[cluster_ID] = cluster_composition
        ret[i] = ret_data
        i += 1
    return ret

# arbitrary definition of cluster type, questionable usefulness...
#
# def sizes_by_cluster_type(cluster_data):
#     '''
#     cluster_data : a list of "snapshot_data", dictionaries by cluster ID whose values are lists of
#     (rod/mol ID, rod state ID) pairs
#     
#     return : a list of "cluster_sizes", dictionaries by cluster type (same as rod state ID if
#     homogeneous, otherwise -1) whose values are pairs of (cluster_sizes, occurrences) lists, and
#     a number equal to the maximum cluster size across all timesteps and types.
#     '''
#     ret = [None]*len(cluster_data)
#     max_size = 0
#     i = 0
#     for snapshot_data in cluster_data:
#         cluster_sizes = {}
#         for cluster in snapshot_data.values():
#             cluster_type = cluster[0][1] # cluster_type is the same as state id of rods if they are all in the same state
#             for elem in cluster:
#                 if elem[1] != cluster_type:
#                     cluster_type = -1
#             
#             cluster_size = len(cluster)
#             try:
#                 cluster_sizes[cluster_type].append(cluster_size)
#             except KeyError:
#                 cluster_sizes[cluster_type] = [cluster_size]
#             
#             if cluster_size > max_size:
#                 max_size = cluster_size
#     
#         for cluster_type, value in cluster_sizes.iteritems():
#             cluster_sizes[cluster_type] = np.unique(value, return_counts=True)
#         
#         ret[i] = cluster_sizes
#         i += 1
#           
#     return ret, max_size

def free_rods(cluster_data, monomer_states=None, total=True):
    '''
    cluster_data : a list of "snapshot_data", dictionaries by cluster ID whose values are lists of
    (rod/mol ID, rod state ID) pairs
    
    monomer_states : a list of rod state ID's, only monomers in those states will be counted;
        if None will count monomers in any state
    
    return : a list of numbers of free monomers (of any type) in each of snapshot_data. If
    total=True a list of (free monomers, total monomers) pairs is returned instead
    '''
    ret = [None]*len(cluster_data)
    i = 0
    for snapshot_data in cluster_data:
        free_monomers = 0
        total_monomers = 0
        for cluster in snapshot_data.values():
            cluster_size = len(cluster)
            total_monomers += cluster_size
            if cluster_size == 1:
                if monomer_states == None or cluster[0][1] in monomer_states:
                    free_monomers += 1
        if total:
            ret[i] = (free_monomers, total_monomers)
        else:
            ret[i] = free_monomers
        i += 1
    
    return ret
