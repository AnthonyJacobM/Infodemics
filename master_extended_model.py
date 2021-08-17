# import relevant libraries
import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from generate_a_bifurcation import generate_a_bifurcation
from generate_protection_bifurcation import generate_protection_bifurcation
from generate_education_bifurcation import generate_education_bifurcation
from generate_risk_bifurcation import generate_risk_bifurcation
from generate_misinformation_bifurcation import generate_misinformation_bifurcation
from generate_infection_good_bifurcation import generate_infection_good_bifurcation
from generate_infection_bad_bifurcation import generate_infection_bad_bifurcation
from generate_recovery_bifurcation import generate_recovery_bifurcation
from plot_nullclines import plot_nullclines
from get_data import get_data
from plot_time_perturbed_steady_states import plot_time_perturbed_steady_state as plot_time_perturbed_steady_states
from generate_limit_cycle_boundary import generate_limit_cycle_boundary
from plot_risk_bifurcation import plot_risk_bifurcation
from plot_protection_bifurcation import plot_protection_bifurcation
from plot_misinformation_bifurcation import plot_misinformation_bifurcation
from plot_education_bifurcation import plot_education_bifurcation
from plot_infection_bad_bifurcation import plot_infection_bad_bifurcation
from plot_infection_good_bifurcation import plot_infection_good_bifurcation
from plot_recovery_bifurcation import plot_recovery_bifurcation
from generate_heat_map_limit_cycle import generate_heat_map_limit_cycle
from plot_limit_cycle_heatmap import plot_limit_cycle_heatmap
from plot_lc_heatmap_revised import plot_lc_heatmap
from plot_a_bifurcation import plot_a_bifurcation
from gen_sys import gen_sys

# get a path associated to saving each figure
path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'
# change to a global variable

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

# Master code to run for all the bifurcations
# set up an initialization for parameters ans initial conditions

# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
                'risk': 0.1, 'protection': 0.90,
                'education': 0.33, 'misinformation': 0.10,
                'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.499, 'x2': 0.50,
                'x3': 0.0001, 'x4': 0.0,
                'x5': 0.50}


# generate a time series
z, t = gen_sys(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = 10_000)

# unpack the parameters
sg, sb, ib, v, phi = z.T

ig = 1 - (sg + sb + v + ib)

# begin plotting
plt.plot(t, sg, 'b')
plt.plot(t, ig, 'b--')
plt.plot(t, sb, 'r')
plt.plot(t, ib, 'r--')
plt.plot(t, v, 'k')
plt.plot(t, phi, 'k--')
plt.xlabel('t (days)')
plt.ylabel('fraction')
plt.show()

# generate new bifurcations

ss_dict_def = {'x1': sg[-1],
               'x2': sb[-1],
               'x3': ib[-1],
               'x4': v[-1],
               'x5': phi[-1]}

# generate an ode
# now we go for it all!
ode = generate_ode(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = 20_000)
# use the dictionary to plot a bifurcation in a
PC_a, par_dict, ss_dict, data = generate_a_bifurcation(ode, ics_dict = ics_dict_def, par_dict = par_dict_def, tend = 1000)