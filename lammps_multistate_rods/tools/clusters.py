# encoding: utf-8
'''
This module provides methods to analyse data from a LAMMPS dump file of a simulation
of rods from the lammps_multistate_rods library in terms of rod clusters.

Created on 16 May 2018

@author: Eugen Rožić
'''

import numpy as np
import re
from parsing import keyword_parse_pattern
from lammps_multistate_rods import Simulation

def get_cluster_data(raw_data, every, model, type_offset):
    '''
    Extracts cluster data from the raw dump file data.
    
    raw_data : output of the "parse_dump_file" method
    
    every : every which snapshot to analyse (e.g. 10 means 9 will be skipped after each analysed)
    
    model : the lammps_multistate_rods.Model class instance which was used to generate the data
    
    type_offset : the type offset for the rod model in the simulation that generated the dump file
    
    returns : a triplet of box dimensions, a list of timesteps and a corresponding list of
    snapshot_data, where "snapshot_data" is a dictionary by cluster ID's whose values are lists of
    (rod/mol ID, rod state ID) pairs
    
    NOTE: cluster ID's are set to the lowest rod/mol ID of each cluster, not the ones that are in
    the dump file
    '''
    states_types = [ [ elem + type_offset 
                        for patch in state_struct for elem in patch]
                            for state_struct in model.state_structures]
    
    def state_types_to_id(state_types):
        '''
        Returns rod state id from its structure (list of atom types)
        '''
        for i in range(model.num_states):
            if state_types == states_types[i]:
                return i
        return None
    
    count = 0
    box_size = None
    timesteps = []
    cluster_data = []
    for timestep, box_bounds, data_structure, data in raw_data:
        count += 1
        if (count-1) % every != 0:
            continue
        timesteps.append(timestep)
        if box_size == None:
            box_size = map(lambda x: x[1]-x[0], box_bounds)
        parse_pattern = re.compile(' '.join(map(keyword_parse_pattern, data_structure)))
        line_vars = {}
        snapshot_data = {}
        current_mol_id = None
        current_cluster_id = None
        current_rod = []
        for line in data: 
            for key, value in zip(data_structure, parse_pattern.match(line).groups()):
                line_vars[key] = value
            mol_id = int(line_vars['mol'])
            cluster_id = int(line_vars['c_'+Simulation.cluster_compute])
            if current_mol_id is None:
                current_mol_id = mol_id
            elif mol_id != current_mol_id:
                current_rod_state = state_types_to_id(current_rod)
                if current_rod_state != None:
                    try:
                        snapshot_data[current_cluster_id].append((current_mol_id, current_rod_state))
                    except KeyError:
                        snapshot_data[current_cluster_id] = [(current_mol_id, current_rod_state)]
                else:
                # means this molecule is not a lammps_multistate_rod "rod"
                # this also takes care of mol_id = 0 (for non-molecule particles)
                    pass
                current_mol_id = mol_id
                current_cluster_id = 0
                current_rod = []
            current_rod.append(int(line_vars['type']))
            if cluster_id > current_cluster_id:
                current_cluster_id = cluster_id    
        current_rod_state = state_types_to_id(current_rod)
        if current_rod_state != None:
            try:
                snapshot_data[current_cluster_id].append((current_mol_id, current_rod_state))
            except KeyError:
                snapshot_data[current_cluster_id] = [(current_mol_id, current_rod_state)]
        else:
            pass
                
        #switch keys to correspond to lowest mol_id in each cluster
        new_snapshot_data = {}
        for value in snapshot_data.values():
            new_snapshot_data[value[0][0]] = value
        cluster_data.append(new_snapshot_data)
    
    return box_size, timesteps, cluster_data

def write_cluster_data(box_size, timesteps, cluster_data, output_path):
    '''
    Writes the data returned by "get_cluster_data" to a file.
    '''
    with open(output_path, 'w') as out_file:
        out_file.write('{:f} {:f} {:f}\n'.format(*box_size))
        for timestep, snapshot_data in zip(timesteps, cluster_data):
            out_file.write('{:^10d} | {:s}\n'.format(timestep, str(snapshot_data)))

def read_cluster_data(input_path):
    '''
    Reads what was output with "write_cluster_data".
    
    returns : a triplet of box dimensions, a list of timesteps and a corresponding list
    of snapshot_data, where "snapshot_data" is a dictionary by cluster ID's whose values
    are lists of (rod/mol ID, rod state ID) pairs.
    '''
    timesteps = []
    cluster_data = []
    
    with open(input_path, 'r') as in_file:
        box_size = map(float, in_file.readline().split())
        for line in in_file:
            timestep, snapshot_data = line.split(' | ')
            timesteps.append(int(timestep))
            cluster_data.append(eval(snapshot_data))
            
    return box_size, timesteps, cluster_data
        
#========================================================================================
# some possible ways of analysis of the raw_data extracted from a dump file...
#========================================================================================

def composition_by_states(cluster_data):
    '''
    cluster_data : a list of "snapshot_data", dictionaries by cluster ID whose values are lists of
    (rod/mol ID, rod state ID) pairs
    
    return : a list of dictionaries by cluster ID whose values are lists of numbers of rods of
    each state in each of the clusters
    '''
    ret = [None]*len(cluster_data)
    i = 0
    for snapshot_data in cluster_data:
        ret_data = {}
        for cluster_ID, cluster in snapshot_data.iteritems():
            max_state_ID = max(map(lambda x: x[1], cluster))
            cluster_composition = [0]*max_state_ID
            for elem in cluster:
                cluster_composition[elem[1]] += 1
            ret_data[cluster_ID] = cluster_composition
        ret[i] = ret_data
        i += 1
    return ret

def sizes_by_cluster_type(cluster_data):
    '''
    cluster_data : a list of "snapshot_data", dictionaries by cluster ID whose values are lists of
    (rod/mol ID, rod state ID) pairs
    
    return : a list of "cluster_sizes", dictionaries by cluster type (same as rod state ID if
    homogeneous, otherwise -1) whose values are pairs of (cluster_sizes, occurrences) lists, and
    a number equal to the maximum cluster size across all timesteps and types.
    '''
    ret = [None]*len(cluster_data)
    max_size = 0
    i = 0
    for snapshot_data in cluster_data:
        cluster_sizes = {}
        for cluster in snapshot_data.values():
            cluster_type = cluster[0][1] # cluster_type is the same as state id of rods if they are all in the same state
            for elem in cluster:
                if elem[1] != cluster_type:
                    cluster_type = -1
            
            cluster_size = len(cluster)
            try:
                cluster_sizes[cluster_type].append(cluster_size)
            except KeyError:
                cluster_sizes[cluster_type] = [cluster_size]
            
            if cluster_size > max_size:
                max_size = cluster_size
    
        for cluster_type, value in cluster_sizes.iteritems():
            cluster_sizes[cluster_type] = np.unique(value, return_counts=True)
        
        ret[i] = cluster_sizes
        i += 1
          
    return ret, max_size

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
