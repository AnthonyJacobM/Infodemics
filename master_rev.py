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

# get a path associated to saving each figure
path = r'D:\Users\antho\PycharmProjects\Infodemics\figures'
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

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

# ---- risk
PC_risk, par_dict_risk, ss_dict_risk, data_risk = plot_risk_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H1', tend = 5_000, eps0 = 0.15)


# --- protection
PC_protection, par_dict_protection, ss_dict_protection, data_protection = plot_protection_bifurcation()


# --- misinformation
PC_misinformation, par_dict_misinformation, ss_dict_misinformation, data_misinformation = plot_misinformation_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)


# --- education
PC_education, par_dict_education, ss_dict_education, data_education = plot_education_bifurcation(par_dict = par_dict_misinformation, ics_dict = ss_dict_misinformation,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)



# --- recovery
PC_reocvery, par_dict_recovery, ss_dict_recovery, data_recovery = plot_recovery_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)



# --- infection_good
PC_infection_good, par_dict_infection_good, ss_dict_infection_good, data_infection_good = plot_infection_good_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)



# --- infection_bad
PC_infection_bad, par_dict_infection_bad, ss_dict_infection_bad, data_infection_bad = plot_infection_bad_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)

