import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from generate_education_bifurcation import generate_education_bifurcation
from plot_nullclines import plot_nullclines
from get_data import get_data
from plot_time_perturbed_steady_states import plot_time_perturbed_steady_state as plot_time_perturbed_steady_states

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
                'risk': 0.10, 'protection': 0.90,
                'education': 0.33, 'misinformation': 0.10,
                'infection_good': 0.048, 'infection_bad': 0.37}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
                'x3': 0.01, 'x4': 0.0,
                'x5': 0.50}


# ------- education! -------
# using the dictionaries for the education

# generate a bifurcation!
eq1_h1_ss_new = {'x1': 0.20993799300477495,
                   'x2': 0.48221092545757344,
                   'x3': 0.07073195797121161,
                   'x4': 0.1168184171123022,
                   'x5': 0.00018891967673741587}

eq1_h1_par_dict_new = par_dict_def
eq1_h1_par_dict_new['risk'] = 1.635295791362042

par_dict_default = eq1_h1_par_dict_new
ss_dict_default = eq1_h1_ss_new


def plot_education_bifurcation(par_dict=par_dict_default, ics_dict=ss_dict_default, special_point='H2', tend=5_000, eps0=0.15):
    """
    function to plot codim-1 bifurcation, nullclines, and perturbed time evolution
    :param par_dict: dictionary of parameters used to simulate the bifurcation
    :param ics_dict: dictionary of initial conditions used to simulate the bifurcation
    :param special_point: point to grab the data on the codim-1 bifurcation
    :param tend: final time used for pointset
    :param eps0: value to perturb the steady states as ics in plotting time 
    :return: PC_education the continuation curve that contains all the data
    """
    # generate the ode
    ode = generate_ode(par_dict, ics_dict=ics_dict, tf=500)

    # generate a pointset!
    pts, ss_dict = generate_pointset(ode, save_bool=True)

    # ------------- risk -------------------------
    # generate risk bifurcation!
    PC_education = generate_education_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_education, curve='EQeducation', special_point=special_point, par_dict=par_dict,
                                           par='education')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='education', random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.6,
                    n_bin=200, par='education')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.50, yhigh=0.5,
                    n_bin=200, par='education')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.75,
                    n_bin=200, par='education')

    return PC_education, par_dict, ss_dict, data



