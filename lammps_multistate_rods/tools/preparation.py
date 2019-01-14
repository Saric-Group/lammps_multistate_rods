# encoding: utf-8
'''
This module contains methods to generate files with locations and rotations of rods
which can then be used to generate rods via the "create_rods" method of a
lammps_multistate_rods.Simulation class instance.

Created on 8 Jan 2019

@author: Eugen Rožić
'''
import numpy as np
import pyquaternion
import os

def fibril(model, N, phi, theta, r0, output_path):
    '''
    This method outputs a file to "output" with data to create a preformed "fibril",
    i.e. a parallel stack of rods.
    
    model : A lammps_multistate_rods.Model clas instance
    phi : azimuth angle (with respect to y-axis; in deg)
    theta : elevation angle (with respect to x-y plane; in deg)
    r0 : centre of the fibril (a triplet)
    output_path : path/name of the output file
    
    returns : a triplet of (min,max) coordinate pairs, one for each dimension, of
    extremal locations among all centers of the rods
    '''    
    rod_radius = model.rod_radius
    if len(r0) == 3:
        r0 = np.array(r0)
    else:
        raise Exception("Center of the fibril has to be a 3-vector!")

    R_z = pyquaternion.Quaternion(axis=[0,0,1], degrees=phi)
    R_x = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta)
    if theta > 0:
        R_x_inv = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta-180)
    else:
        R_x_inv = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta+180)

    # correct composite rotation is to first rotate around z then around x', which is equivalent to
    # rotations first around x then around z for the same angles
    R_tot = R_z * R_x
    R_tot_inv = R_z * R_x_inv

    locations = [None]*N # list of position 3-vectors
    rotations = [None]*N # list of tuples of a position 3-vector and angle
    mins = [np.infty]*3
    maxs = [-np.infty]*3

    for i in range(N):
        if i % 2 == 0:
            locations[i] = np.array([0, (i-N/2)*rod_radius, -rod_radius])#+0.27692])
            locations[i] = R_tot.rotate(locations[i]) + r0
            if R_tot.angle == 0.0:
                rotations[i] = (R_tot.angle, [1.0, 0.0, 0.0])
            else:
                rotations[i] = (R_tot.angle, R_tot.axis)
        else:
            locations[i] = np.array([0, (i-N/2)*rod_radius, +rod_radius])#-0.27692])
            locations[i] = R_tot.rotate(locations[i]) + r0
            if R_tot_inv.angle == 0.0:
                rotations[i] = (R_tot_inv.angle, [1.0, 0.0, 0.0])
            else:
                rotations[i] = (R_tot_inv.angle, R_tot_inv.axis)
        
        for j in range(3):
            if locations[i][j] > maxs[j]:
                maxs[j] = locations[i][j]
            if locations[i][j] < mins[j]:
                mins[j] = locations[i][j]
    
    if os.path.exists(output_path):
        print "WARNING: {:s} already exists, won't overwrite it...".format(output_path)
        return zip(mins, maxs)

    with open(output_path, 'w') as output_file:
        output_file.write('monomers: {:d}\n\n'.format(N))
        for loc, rot in zip(locations, rotations):
            output_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.3f} {:.3f}\n'.format(
                loc[0], loc[1], loc[2], rot[0], *rot[1]))
        output_file.write('\n')
    
    return zip(mins, maxs)

def single(r0, phi, theta, output_path):
    '''
    This method outputs a file to "output" with data to create a single rod at the given
    location and with the given orientation.
    
    r0 : centre of the rod (a triplet)
    phi : azimuth angle (with respect to x-axis; in deg)
    theta : elevation angle (with respect to x-y plane; in deg)
    output_path : path/name of the output file
    '''
    
    R_z = pyquaternion.Quaternion(axis=[0,0,1], degrees=phi)
    R_x = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta)
    R_tot = R_z * R_x
    
    if len(r0) == 3:
        r0 = np.array(r0)
    else:
        raise Exception('Center of the rod has to be a 3-vector!')

    with open(output_path, 'w') as output_file:
        output_file.write('monomers: 1\n\n')
        output_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.3f} {:.3f}\n'.format(
            r0[0], r0[1], r0[2], R_tot.angle, *R_tot.axis))
        output_file.write('\n')

def random(model, N, bounds, output_path):
    '''
    Randomly placed and rotated N non-overlapping rods inside given bounds...
    TODO
    '''
    pass #TODO