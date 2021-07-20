import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from generate_protection_bifurcation import generate_protection_bifurcation
from generate_education_bifurcation import generate_education_bifurcation
from generate_risk_bifurcation import generate_risk_bifurcation
from generate_bifurcation_2d import generate_bifurcation_2d
from plot_nullclines import plot_nullclines_new as plot_nullclines
from get_data import get_data
from plot_time_perturbed_steady_states import plot_time_perturbed_steady_state as plot_time_perturbed_steady_states

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
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

# generate an ode

# generate the ode
ode = generate_ode(eq1_h1_par_dict, ics_dict = eq1_h1_ss)

# generate a pointset!
pts, ss_dict = generate_pointset(ode, save_bool = True)

# generate risk bifurcation!
PC = generate_risk_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, tend = 300)

# get the data
par_dict, ss_dict, data = get_data(PC, curve = 'EQrisk', special_point = 'H1', par_dict = eq1_h1_par_dict)
# use EQrisk2 and BP1 (see generate_risk_bifurcation) for more details!

# generate a pointset and plot the time around the bifurcation!
pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = 10_000, par = 'risk', random_bool = True, eps = 0.01)


# plot a few nulllclines
# sg vs sb nullcline
plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200)
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200)
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200)



