# encoding: utf-8
'''
An interactive console application for drawing interactive plots of interactions
between rods from a lammps_multistate_rods Rod_model defined in a config file.

Created on 30 Nov 2018

@author: Eugen Rožić
'''

import numpy as np

from lammps_multistate_rods import Rod_model
import lammps_multistate_rods.tools.interactions as md

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def draw_models(axes = None, r = 0, z = 0, phi = 0, old = []):
    
    if axes is None :
        axes = plt.gca()
    r_body = md.model.rod_radius
    r_int = md.model.patch_bead_radii
    
    for i in range(len(old),0,-1):
        old[i-1].remove()
        del old[i-1]
            
    new = old
    
    # the MD model body
    for i in range(md.M[0]): 
        new.append(
            axes.add_patch(plt.Circle((z + md.bead_z[0][i], r), 2*r_body, fill=True, color='#DDDDDD', alpha=0.7)))
    for i in range(md.M[0]):
        new.append(
            axes.add_patch(plt.Circle((z + md.bead_z[0][i], r), r_body, fill=True, color='#777777')))
    
    # the MD model interaction centers (patches)
    for k in range(1, md.K):
        r_k = r + md.patch_r[k]*np.cos(md.patch_phi[k] + phi)
        for i in range(md.M[k]):
            new.append(
                axes.add_patch(plt.Circle((z + md.bead_z[k][i], r_k), r_int[k-1], fill=True, color='red')))
    
    return new

def draw_point_rod_2D(vals, vals_min, fig_title = ''):
    
    fig_2D, ax_2D = plt.subplots(num=fig_title, figsize=(10,7))
    fig_2D.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.99) #0.86 x 0.86
    
    img = ax_2D.imshow(vals[0], extent=[zmin, zmax, rmin, rmax],
                       vmin=vals_min, vmax=-vals_min,
                       origin="lower", interpolation='bilinear', cmap=img_cmap)
    model_patches = draw_models(ax_2D)
    
    ax_2D.set_xlabel(r'$z$', **axis_font)
    ax_2D.set_ylabel(r'$r$', **axis_font)
    ax_2D.axis([zmin, zmax, rmin, rmax])
    
    phi_slider_axes = fig_2D.add_axes([0.40, 0.02, 0.30, 0.03], facecolor=widget_color)
    phi_slider = Slider(phi_slider_axes, r'$\phi$', 0, theta_points-1, valinit=zero_theta,
                        valstep=1, valfmt = '%1.1f deg')
    
    def update_img(_):
        curr_phi_index = int(phi_slider.val)
        curr_phi_val = thetas[curr_phi_index]
        phi_slider.valtext.set_text(phi_slider.valfmt % (np.rad2deg(curr_phi_val)))
        draw_models(ax_2D, phi = curr_phi_val, old = model_patches)
        img.set_data(vals[curr_phi_index])
        fig_2D.canvas.draw_idle()
    
    phi_slider.on_changed(update_img)
    
    return phi_slider

def draw_point_rod_z_slice(vals, vals_min, fig_title = ''):
    # 1D r-E plot
    fig_r, ax_r = plt.subplots(num=fig_title, figsize=(9,5))
    fig_r.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.99) #0.86 x 0.86
    
    lines = []
    for i in range(theta_points):
        lines.append(ax_r.plot(rs, vals[i].T[zero_z], 'r-', lw=1.0,
                               label=r'$\phi = {:1.1f}^\circ$'.format(np.rad2deg(thetas[i])),
                               color = plot_cmap((i+1.0)/(theta_points+1))))
    ax_r.axvline(2.0*model.rod_radius, color='black', linestyle='-', lw=1.0)
    ax_r.axvline(3.0*model.rod_radius, color='black', linestyle='--', lw=0.5)
    ax_r.grid()
    
    ax_r.set_xlabel(r'$r$', **axis_font)
    ax_r.set_ylabel(r'$E$', **axis_font)
    ax_r.legend(loc = 'lower left', prop = axis_font)
    ax_r.axis([rmin, rmax, 1.05*vals_min, -1.05*vals_min])
    
    z_slider_axes = fig_r.add_axes([0.4, 0.02, 0.3, 0.03], facecolor=widget_color)
    z_slider = Slider(z_slider_axes, r'$z$', 0, z_points-1, valinit=zero_z,
                      valstep=1, valfmt = '%1.2f')
    
    def update_plot(_):
        curr_z_index = int(z_slider.val)
        curr_z_val = zs[curr_z_index]
        z_slider.valtext.set_text(z_slider.valfmt % curr_z_val)
        for i in range(theta_points):
            lines[i][0].set_ydata(vals[i].T[curr_z_index])
        fig_r.canvas.draw_idle()
    
    z_slider.on_changed(update_plot)
    
    return z_slider

def draw_point_rod_r_slice(vals, vals_min, fig_title = ''):
    # 1D z-E plot
    fig_z, ax_z = plt.subplots(num=fig_title, figsize=(10,5))
    fig_z.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.99) #0.86 x 0.86
    
    r_init = int((model.rod_radius*2 - rmin)/dx)
    lines = []
    for i in range(theta_points):
        lines.append(ax_z.plot(zs, vals[i][r_init], 'r-', lw=1.0,
                               label=r'$\phi = {:1.1f}^\circ$'.format(np.rad2deg(thetas[i])),
                               color = plot_cmap((i+1.0)/(theta_points+1))))
    ax_z.axvline(-3.0*model.rod_radius, color='black', linestyle='--', lw=1.0)
    ax_z.axvline(3.0*model.rod_radius, color='black', linestyle='--', lw=1.0)
    ax_z.grid()
    
    ax_z.set_xlabel(r'$z$', **axis_font)
    ax_z.set_ylabel(r'$E$', **axis_font)
    ax_z.legend(loc = 'lower right', prop = axis_font)
    ax_z.axis([zmin, zmax, 1.05*vals_min, -1.05*vals_min])
    
    r_slider_axes = fig_z.add_axes([0.4, 0.02, 0.3, 0.03], facecolor=widget_color)
    r_slider = Slider(r_slider_axes, r'$r$', 0, r_points-1, valinit=r_init,
                      valstep=1, valfmt = '%1.2f')
    r_slider.valtext.set_text(r_slider.valfmt % (r_init*dx))
    
    def update_plot(_):
        curr_r_index = int(r_slider.val)
        curr_r_val = rs[curr_r_index]
        r_slider.valtext.set_text(r_slider.valfmt % curr_r_val)
        for i in range(theta_points):
            lines[i][0].set_ydata(vals[i][curr_r_index])
        fig_z.canvas.draw_idle()
    
    r_slider.on_changed(update_plot)
    
    return r_slider

#========================================================================================

def draw_rod_rod_2D(vals, vals_min, fig_title = ''):
    
    fig_2D, ax_2D = plt.subplots(num=fig_title, figsize=(10,7))
    fig_2D.subplots_adjust(left=0.1, bottom=0.20, right=0.99, top=0.99) #0.88 x 0.88
    
    if len(vals) > 1:
        psi2_init = zero_phi
    else:
        psi2_init = 0
    img = ax_2D.imshow(vals[psi2_init][zero_theta],
                    extent=[zmin, zmax, rmin, rmax], vmin=vals_min, vmax=-vals_min,
                    origin="lower", interpolation='bilinear', cmap=img_cmap)
    model_patches = draw_models(ax_2D)
    
    ax_2D.set_xlabel(r'$z$', **axis_font)
    ax_2D.set_ylabel(r'$r$', **axis_font)
    ax_2D.axis([zmin, zmax, rmin, rmax])
    
    psi1_slider = psi2_slider = None
    
    if len(vals) > 1:
        psi2_slider_axes = fig_2D.add_axes([0.4, 0.06, 0.3, 0.03], facecolor=widget_color)
        psi2_slider = Slider(psi2_slider_axes, r'$\psi_2$', 0, phi_points-1, valinit=zero_phi,
                        valstep=1, valfmt = '%1.1f deg')
        psi2_slider.valtext.set_text(psi2_slider.valfmt % 0.0)
        
    if len(vals[0]) > 1:
        psi1_slider_axes = fig_2D.add_axes([0.4, 0.02, 0.3, 0.03], facecolor=widget_color)
        psi1_slider = Slider(psi1_slider_axes, r'$\psi_1$', 0, theta_points-1, valinit=zero_theta,
                        valstep=1, valfmt = '%1.1f deg')
    
    def update_img(_):
        curr_psi1_index = curr_psi2_index = 0
        if psi1_slider != None:
            curr_psi1_index = int(psi1_slider.val)
            curr_psi1_val = thetas[curr_psi1_index]
            psi1_slider.valtext.set_text(psi1_slider.valfmt % (np.rad2deg(curr_psi1_val)))
            draw_models(ax_2D, phi = curr_psi1_val, old = model_patches)
        if psi2_slider != None:
            curr_psi2_index = int(psi2_slider.val)
            curr_psi2_val = phis[curr_psi2_index]
            psi2_slider.valtext.set_text(psi2_slider.valfmt % (np.rad2deg(curr_psi2_val)))
        img.set_data(vals[curr_psi2_index][curr_psi1_index])
        fig_2D.canvas.draw_idle()
    
    if psi1_slider != None:
        psi1_slider.on_changed(update_img)
    if psi2_slider != None:
        psi2_slider.on_changed(update_img)
        
        return psi1_slider, psi2_slider

def draw_rod_rod_z_slice(vals, vals_min, fig_title = ''):
    # 1D r-E plot
    fig_r, ax_r = plt.subplots(num=fig_title, figsize=(9,6))
    fig_r.subplots_adjust(left=0.1, bottom=0.20, right=0.99, top=0.99) #0.86 x 0.86
    
    if len(vals) > 1:
        psi2_init = zero_phi
    else:
        psi2_init = 0
    lines = []
    for i in range(theta_points):
        lines.append(ax_r.plot(rs, vals[psi2_init][i].T[zero_z], 'r-', lw=1.0,
                               label=r'$\psi_1 = {:1.1f}^\circ$'.format(np.rad2deg(thetas[i])),
                                   color = plot_cmap((i+1.0)/(theta_points+1))))
    ax_r.axvline(2.0*model.rod_radius, color='black', linestyle='-', lw=1.0)
    ax_r.axvline(3.0*model.rod_radius, color='black', linestyle='--', lw=0.5)
    ax_r.grid()
    
    ax_r.set_xlabel(r'$r$', **axis_font)
    ax_r.set_ylabel(r'$E$', **axis_font)
    ax_r.legend(loc = 'lower left', prop = axis_font)
    ax_r.axis([rmin, rmax, 1.05*vals_min, -1.05*vals_min])
    
    psi2_slider = None
    
    if len(vals) > 1:
        psi2_slider_axes = fig_r.add_axes([0.4, 0.06, 0.3, 0.03], facecolor=widget_color)
        psi2_slider = Slider(psi2_slider_axes, r'$\psi_2$', 0, phi_points-1, valinit=zero_phi,
                        valstep=1, valfmt = '%1.1f deg')
        psi2_slider.valtext.set_text(psi2_slider.valfmt % 0.0)
    
    z_slider_axes = fig_r.add_axes([0.4, 0.02, 0.3, 0.03], facecolor=widget_color)
    z_slider = Slider(z_slider_axes, r'$z$', 0, z_points-1, valinit=zero_z,
                      valstep=1, valfmt = '%1.2f')
    
    def update_plot(_):
        if psi2_slider != None:
            curr_psi2_index = int(psi2_slider.val)
            curr_psi2_val = phis[curr_psi2_index]
            psi2_slider.valtext.set_text(psi2_slider.valfmt % (np.rad2deg(curr_psi2_val)))
        else:
            curr_psi2_index = 0
        curr_z_index = int(z_slider.val)
        curr_z_val = zs[curr_z_index]
        z_slider.valtext.set_text(z_slider.valfmt % curr_z_val)
        for i in range(theta_points):
            lines[i][0].set_ydata(vals[curr_psi2_index][i].T[curr_z_index])
        fig_r.canvas.draw_idle()
    
    if psi2_slider != None:
        psi2_slider.on_changed(update_plot)
    z_slider.on_changed(update_plot)
    
    return psi2_slider, z_slider

def draw_rod_rod_r_slice(vals, vals_min, fig_title = ''):
    # 1D z-E plot
    fig_z, ax_z = plt.subplots(num=fig_title, figsize=(10,6))
    fig_z.subplots_adjust(left=0.1, bottom=0.20, right=0.99, top=0.99) #0.86 x 0.86
    
    if len(vals) > 1:
        psi2_init = zero_phi
    else:
        psi2_init = 0
    r_init = int((model.rod_radius*2 - rmin)/dx)
    lines = []
    for i in range(theta_points):
        lines.append(ax_z.plot(zs, vals[psi2_init][i][r_init], 'r-', lw=1.0,
                               label=r'$\psi_1 = {:1.1f}^\circ$'.format(np.rad2deg(thetas[i])),
                                   color = plot_cmap((i+1.0)/(theta_points+1))))
    ax_z.axvline(-3.0*model.rod_radius, color='black', linestyle='--', lw=1.0)
    ax_z.axvline(3.0*model.rod_radius, color='black', linestyle='--', lw=1.0)
    ax_z.grid()
    
    ax_z.set_xlabel(r'$z$', **axis_font)
    ax_z.set_ylabel(r'$E$', **axis_font)
    ax_z.legend(loc = 'lower right', prop = axis_font)
    ax_z.axis([zmin, zmax, 1.05*vals_min, -1.05*vals_min])
    
    psi2_slider = None
    
    if len(vals) > 1:
        psi2_slider_axes = fig_z.add_axes([0.4, 0.06, 0.3, 0.03], facecolor=widget_color)
        psi2_slider = Slider(psi2_slider_axes, r'$\psi_2$', 0, phi_points-1, valinit=zero_phi,
                        valstep=1, valfmt = '%1.1f deg')
        psi2_slider.valtext.set_text(psi2_slider.valfmt % 0.0)
    
    r_slider_axes = fig_z.add_axes([0.4, 0.02, 0.3, 0.03], facecolor=widget_color)
    r_slider = Slider(r_slider_axes, r'$r$', 0, r_points-1, valinit=r_init,
                      valstep=1, valfmt = '%1.2f')
    r_slider.valtext.set_text(r_slider.valfmt % (r_init*dx))
    
    def update_plot(_):
        if psi2_slider != None:
            curr_psi2_index = int(psi2_slider.val)
            curr_psi2_val = phis[curr_psi2_index]
            psi2_slider.valtext.set_text(psi2_slider.valfmt % (np.rad2deg(curr_psi2_val)))
        else:
            curr_psi2_index = 0
        curr_r_index = int(r_slider.val)
        curr_r_val = rs[curr_r_index]
        r_slider.valtext.set_text(r_slider.valfmt % curr_r_val)
        for i in range(theta_points):
            lines[i][0].set_ydata(vals[curr_psi2_index][i][curr_r_index])
        fig_z.canvas.draw_idle()
    
    if psi2_slider != None:
        psi2_slider.on_changed(update_plot)
    r_slider.on_changed(update_plot)
        
    return psi2_slider, r_slider

#======================================================================================

def calculate_point_rod(bead_type, rod_state):
    
    vals = np.array([md.bead_rod_interaction(bead_type, rod_state, r, z, psi1)
                              for psi1 in thetas for r in rs for z in zs])
    vals = vals.reshape(theta_points, r_points, z_points)
    vals_min = vals.min()
    print("min val = {}".format(vals_min))
    
    return vals, vals_min
    
def calculate_rod_rod(rod1_state, rod2_state, theta = 0, phi = 0, psi2 = None):
    '''
    theta : polar angle of rod at (r,z)
    phi : azimuthal angle of rod at (r,z)
    psi2 : angle of patch of rod at (r,z); if "None" calculated for psi2 in [-pi,pi>
    '''
    if psi2 != None:
        psi2s = [psi2]
        psi2_points = 1
    else:
        psi2s = phis
        psi2_points = phi_points
    
    vals = np.array([md.rod_rod_interaction(rod1_state, rod2_state, r, z, theta, phi, psi1, psi2)
                              for psi2 in psi2s for psi1 in thetas for r in rs for z in zs])
    vals = vals.reshape(psi2_points, theta_points, r_points, z_points)
    vals_min = vals.min()
    print("min val = {}".format(vals_min))
    
    return vals, vals_min

#=======================================================================================

import argparse

parser = argparse.ArgumentParser(description='An application for interactive visualisation'\
                                 ' of interaction potentials for rod models of the "lammps_'\
                                 'multistate_rods" library',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('config_file', type=str,
                    help='path to the "lammps_multistate_rods" model config file')
parser.add_argument('--rmin', type=float, default=0.0,
                        help='lower bound for the "r" variable')
parser.add_argument('--rmax', type=float, default=5.0,
                        help='upper bound for the "r" variable')
parser.add_argument('--zmin', type=float, default=0.0,
                        help='lower bound for the "z" variable')
parser.add_argument('--zmax', type=float, default=7.5,
                        help='upper bound for the "z" variable')
parser.add_argument('--dx', type=float, default=0.1,
                        help='the grid spacing (in both "r" and "z")')
parser.add_argument('--da', type=float, default=30,
                        help='the angle step (for all angles; in deg)')
args = parser.parse_args()

if __name__ != '__main__':
    print("(visualise.py) ERROR: This module should only be called as a stand-alone application!")
    quit()

cfg_filename = args.config_file
model = Rod_model(cfg_filename)
md.model_setup(model)

# plot parameters and grid points
dx = args.dx*model.rod_radius
da = np.deg2rad(args.da)

rmin = args.rmin*model.rod_radius; rmax = args.rmax*model.rod_radius
r_points = int((rmax-rmin)/dx) + 1; zero_r = 0
rs = np.linspace(rmin, rmax, r_points)

zmin = args.zmin*model.rod_radius; zmax = args.zmax*model.rod_radius
z_points = int((zmax-zmin)/dx) + 1; zero_z = 0
zs = np.linspace(zmin, zmax, z_points)

phimin = -np.pi; phimax = np.pi
phi_points = int((phimax-phimin)/da); zero_phi = phi_points/2
phis = np.linspace(phimin, phimax, phi_points, endpoint=False)

thetamin = 0; thetamax = np.pi
theta_points = int((thetamax-thetamin)/da) + 1; zero_theta = 0
thetas = np.linspace(thetamin, thetamax, theta_points)

# figure parameters
axis_font = {'size':13}
img_cmap = plt.get_cmap("RdBu")
plot_cmap = plt.cm.get_cmap('nipy_spectral')
widget_color = 'lightgoldenrodyellow'

# interactive console - choice of interaction
while True:
    print()
    int_type = input("Enter '1' for point-rod or '2' for rod-rod interaction: ")
    int_type = int_type.strip()
    if int_type == '1':
        while True:
            bead_type = input("Enter type of bead at (z,r): ")
            try:
                bead_type = int(bead_type)
                if bead_type not in model.all_bead_types:
                    raise Exception("")
                break
            except:
                print("Unexpected input ({:}), please enter one of the following: ".format(bead_type),
                      model.all_bead_types)
        for i in range(model.num_states):
                print("{:2d} : {:s}".format(i, model.rod_states[i]))
        while True:
            rod_state = input("Enter state ID of rod at (0,0): ")
            try:
                rod_state = int(rod_state)
                if rod_state < 0 or rod_state >= model.num_states:
                    raise Exception("")
                break
            except:
                print("Unexpected input ({:}), please try again...".format(rod_state))
        print("Calculating...")
        vals, vals_min = calculate_point_rod(bead_type, rod_state)
        break
    elif int_type == '2':
        for i in range(model.num_states):
            print("{:2d} : {:s}".format(i, model.rod_states[i]))
        while True:
            rod1_state = input("Enter state ID of rod at (0,0): ")
            try:
                rod1_state = int(rod1_state)
                if rod1_state < 0 or rod1_state >= model.num_states:
                    raise Exception("")
                break
            except:
                print("Unexpected input ({:}), please try again...".format(rod1_state))
        while True:
            rod2_state = input("Enter state ID of rod at (z,r): ")
            try:
                rod2_state = int(rod2_state)
                if rod2_state < 0 or rod2_state >= model.num_states:
                    raise Exception("")
                break
            except:
                print("Unexpected input ({:}), please try again...".format(rod2_state))
        while True:
            theta = input("Enter angle from z axis (theta; in deg) of rod at (z,r): ")
            if theta.strip() == '':
                theta = 0
                break
            else:
                try:
                    theta = np.deg2rad(float(theta))
                    break
                except:
                    print("Unexpected input ({:}), please try again...".format(theta))
        while True:
            phi = input("Enter angle around z axis (phi; in deg) of rod at (z,r): ")
            if phi.strip() == '':
                phi = 0
                break
            else:
                try:
                    phi = np.deg2rad(float(phi))
                    break
                except:
                    print("Unexpected input ({:}), please try again...".format(phi))
        while True:
            psi2 = input("Enter internal rotation (psi2; in deg) of rod at (z,r): ")
            if psi2.strip() == '':
                psi2 = None
                break
            else:
                try:
                    psi2 = np.deg2rad(float(psi2))
                    break
                except:
                    print("Unexpected input ({:}), please try again...".format(psi2))
        print("Calculating...")
        vals, vals_min = calculate_rod_rod(rod1_state, rod2_state, theta, phi, psi2)
        break
    else:
        print("Unexpected input ({:s}), please try again...".format(int_type))

# interactive console - choice of plot
while True:
    print("""
1 : 2D plot
2 : z-slice plot
3 : r-slice plot
q : Quit""")
    plot_type = input("What to plot? (enter 1, 2 or 3) ")
    plot_type = plot_type.strip()
    if plot_type == '1':
        if int_type == '1':
            widgets = draw_point_rod_2D(vals, vals_min,
                                        "point({:d})-rod({:s}) interaction  ({:s})".format(
                                            bead_type, model.rod_states[rod_state],
                                            cfg_filename))
        else:
            widgets = draw_rod_rod_2D(vals, vals_min,
                                      "rod({:s})-rod({:s}) interaction  ({:s})".format(
                                            model.rod_states[rod1_state],
                                            model.rod_states[rod2_state],
                                            cfg_filename))
    elif plot_type == '2':
        if int_type == '1':
            widgets = draw_point_rod_z_slice(vals, vals_min,
                                        "point({:d})-rod({:s}) interaction  ({:s})".format(
                                            bead_type, model.rod_states[rod_state],
                                            cfg_filename))
        else:
            widgets = draw_rod_rod_z_slice(vals, vals_min,
                                      "rod({:s})-rod({:s}) interaction  ({:s})".format(
                                            model.rod_states[rod1_state],
                                            model.rod_states[rod2_state],
                                            cfg_filename))
    elif plot_type == '3':
        if int_type == '1':
            widgets = draw_point_rod_r_slice(vals, vals_min,
                                        "point({:d})-rod({:s}) interaction  ({:s})".format(
                                            bead_type, model.rod_states[rod_state],
                                            cfg_filename))
        else:
            widgets = draw_rod_rod_r_slice(vals, vals_min,
                                      "rod({:s})-rod({:s}) interaction  ({:s})".format(
                                            model.rod_states[rod1_state],
                                            model.rod_states[rod2_state],
                                            cfg_filename))
    elif plot_type.lower() == 'q':
        quit()
    else:
        print("Unexpected input ({:s}), please try again...".format(plot_type))
        continue
    plt.show()
