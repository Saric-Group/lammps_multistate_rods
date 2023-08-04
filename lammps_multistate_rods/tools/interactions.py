# encoding: utf-8
'''
Contains methods that calculate the interaction energy between rods in different states
for a lammps_multistate_rods model.

Created on 12 Apr 2018

@author: Eugen Rožić
'''

from .potentials import lj_n_m
from .potentials import cos_sq
from .potentials import morse
from .potentials import gauss

import numpy as np

model = None

K = None # num of patches (+body)
M = None # nums of patch beads by patch (incl body)

r_rod_sq = None

bead_z = None
patch_r = None
patch_phi = None

sol_active = None
beta_active = None

interactions = {}

def interaction_function(int_type, R, eps):
    if int_type[0] == 'lj/cut':
        return lambda r: lj_n_m(12, 6, r, R, R + int_type[1], eps)
    elif int_type[0] == 'cosine/squared':
        return lambda r: cos_sq(r, R, R + int_type[1], eps,
                                           True if len(int_type)==3 else False)
    elif int_type[0] == 'nm/cut':
        return lambda r: lj_n_m(int_type[1], int_type[2], r, R, R + int_type[3], eps)
    elif int_type[0] == 'morse':
        return lambda r: morse(int_type[1], r, R, R + int_type[2], eps)
    elif int_type[0] == 'gauss/cut':
        return lambda r: gauss(int_type[1], r, R, R + int_type[2], eps)
    else:
        raise Exception('Unknown/invalid int_type parameter: '+ str(int_type))

def model_setup(rod_model):
    '''
    rod_model : a lammps_multistate_rods.Model instance
    '''
    global model
    model = rod_model
    
    global K, M, r_rod_sq
    K = 1 + model.num_patches #body bead as a 0th patch
    M = model.num_beads
    r_rod_sq = model.rod_radius**2
    
    global bead_z, patch_r, patch_phi
    bead_z = [None]*K
    patch_r = [0.0]*K
    patch_phi = [None]*K #so error is thrown if patch_phi[0] is used for calculation...
    
    bead_z[0] = [(i - (M[0]-1)/2.)*(2*model.rod_radius - model.body_bead_overlap)
                 for i in range(M[0])]
    bead_z[1:] = [[(i - (M[k]-1)/2.)*(2*model.patch_bead_radii[k-1] + model.patch_bead_sep[k-1])
                   for i in range(M[k])] for k in range(1,K)]
    patch_r[1:] = [model.rod_radius - model.patch_bead_radii[k-1] + model.patch_bulge_out[k-1]
                   for k in range(1,K)]
    patch_phi[1:] = [np.deg2rad(model.patch_angles[k-1]) for k in range(1,K)]
    
    def radius_from_type(bead_type):
        if bead_type in model.body_bead_types:
            return model.rod_radius
        else:
            for i in range(model.num_patches):
                if bead_type in model.patch_bead_types[i]:
                    return model.patch_bead_radii[i]
    
    for (type1, type2), (eps, int_type_key) in model.eps.iteritems():
        r1 = radius_from_type(type1)
        r2 = radius_from_type(type2)
        interactions[(type1, type2)] = interactions[(type2, type1)] = \
            interaction_function(model.int_types[int_type_key], r1+r2, eps)


def bead_rod_interaction(point_bead_type, rod_state, r, z, phi):
    '''
    Interaction between a rod in the given state centered at (0,0) and extending along
    the z-axis and a bead of the given type centered at (r,z).
    
    phi : the orientation of the rod at (0,0)
    '''
    U = 0
    for k in range(K):
        patch = model.state_structures[rod_state][k]
        if k == 0:
            ort_part = r**2 #saving some calculation
        else:
            r_i = patch_r[k]
            phi_i = phi + patch_phi[k]
            ort_part = r**2 + r_i**2 - 2*r_i*r*np.cos(phi_i)
        for i in range(M[k]):
            bead_type = patch[i]
            try:
                interaction = interactions[(point_bead_type, bead_type)]
            except KeyError:
                continue #no entry means no interaction
            z_i = bead_z[k][i]
            dist = np.sqrt(ort_part + (z - z_i)**2)
            U += interaction(dist)
            if U == float("inf"):
                return 0.0 #instant break if volume exclusion overlap
    return U

def rod_rod_interaction(rod1_state, rod2_state, r, z, theta, phi, psi1, psi2):
    '''
    Interaction between two rods in the given states. The first rod is at (0,0) with internal
    rotation of psi1, and the other is at (z,r) with orientation (theta, phi) and internal
    rotation psi2.
    
    theta : angle from the z axis of the rod at (z,r)
    phi :  angle about the z axis of the rod at (z,r)
    psi1 : the direction of the patch vector of the rod at (0,0)
    psi2 : the direction of the patch vector of the rod at (r,z); the patch starts facing
        down and turns in the opposite direction to psi1 (as if the rods are mirror images)
    '''
    if theta < 0 or theta > np.pi:
        raise Exception("Theta has to be in interval [0, pi]")
    elif theta < np.pi/2:
        psi2 = np.pi - phi - psi2
    else:
        psi2 = phi + psi2
    
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    
    x1 = [0.0]*K
    y1 = [0.0]*K
    z1 = bead_z
    x2 = [None]*K
    y2 = [None]*K
    z2 = [None]*K
    for k in range(K):
        r1 = r2 = patch_r[k]
        if k > 0:
            psi1_k = psi1 + patch_phi[k]
            c_psi1 = np.cos(psi1_k)
            s_psi1 = np.sin(psi1_k)
            psi2_k = psi2 + patch_phi[k]
            c_psi2 = np.cos(psi2_k)
            s_psi2 = np.sin(psi2_k)
            x1[k] = r1*c_psi1
            y1[k] = r1*s_psi1
        x2[k] = [0.0]*M[k]
        y2[k] = [0.0]*M[k]
        z2[k] = [0.0]*M[k]
        for i in range(M[k]):
            if k == 0:
                x2[k][i] = bead_z[k][i]*c_phi*s_theta
                y2[k][i] = bead_z[k][i]*s_phi*s_theta
                z2[k][i] = bead_z[k][i]*c_theta
            else:
                x2[k][i] = r2*(c_psi2*c_phi*c_theta - s_psi2*s_phi) + bead_z[k][i]*c_phi*s_theta
                y2[k][i] = r2*(c_psi2*s_phi*c_theta + s_psi2*c_phi) + bead_z[k][i]*s_phi*s_theta
                z2[k][i] = -r2*c_psi2*s_theta + bead_z[k][i]*c_theta
    
    U = 0
    for k1 in range(K):
        patch1 = model.state_structures[rod1_state][k1]
        for i in range(M[k1]):
            bead1_type = patch1[i]
            for k2 in range(K):
                patch2 = model.state_structures[rod2_state][k2]
                for j in range(M[k2]):
                    bead2_type = patch2[j]
                    try:
                        interaction = interactions[(bead1_type, bead2_type)]
                    except KeyError:
                        continue #no entry means no interaction
                    dist = np.sqrt((r + x2[k2][j] - x1[k1])**2 + 
                                   (y2[k2][j] - y1[k1])**2 +
                                   (z + z2[k2][j] - z1[k1][i])**2)
                    U += interaction(dist)
                    if U == float("inf"):
                        return 0.0 #instant break if volume exclusion overlap
    
    return U
