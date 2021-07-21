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
from generate_bifurcation_2d import generate_bifurcation_2d
from plot_nullclines import plot_nullclines
from get_data import get_data
from plot_time_perturbed_steady_states import plot_time_perturbed_steady_state as plot_time_perturbed_steady_states
from generate_limit_cycle_boundary import generate_limit_cycle_boundary

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

# generate an ode

# generate the ode
ode = generate_ode(eq1_h1_par_dict, ics_dict = eq1_h1_ss, tf=500)

# generate a pointset!
pts, ss_dict = generate_pointset(ode, save_bool = True)


# ------------- risk -------------------------
# generate risk bifurcation!
PC_risk = generate_risk_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, tend = 300)

# get the data
par_dict, ss_dict, data = get_data(PC_risk, curve = 'EQrisk2', special_point = 'BP1', par_dict = eq1_h1_par_dict, par = 'risk')
# use EQrisk2 and BP1 (see generate_risk_bifurcation) for more details!

# generate a pointset and plot the time around the bifurcation!
pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = 5_000, par = 'risk', random_bool = True, eps = 0.1)


# plot a few nulllclines
# sg vs sb nullcline
plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'risk')
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'risk')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'risk')

# generate bifurcation for protection eg. vaccination efficacy
# ----------- protection ---------------------
# generate risk bifurcation!

# dictionaries for parameters obtained on the LP1 previously!
eq1_risk_lp1_par_dict = {'risk':  0.12952930209570576, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

eq1_risk_lp1_ss = { 'x1':  0.02179992969509327,
                    'x2':  0.21186033176092678,
                    'x3':  0.07439263859553721,
                    'x4':  0.6784593698650656,
                    'x5':  0.011113221022650485}


# use these for the bifurcation in delta, vaccination efficacy!
par_dict = eq1_risk_lp1_par_dict
ss_dict = eq1_risk_lp1_ss

# compute bifurcation in delta!
PC_protection = generate_protection_bifurcation(ode, ics_dict = ss_dict, par_dict = par_dict, tend = 300, max_points = 120)

# get the data
par_dict, ss_dict, data = get_data(PC_protection, curve = 'EQprotection', special_point = 'LP1', par_dict = par_dict, par = 'protection')

# generate a pointset and plot the time around the bifurcation!
pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = 5_000, par = 'protection', random_bool = True, eps = 0.05)


# plot a few nulllclines
# sg vs sb nullcline
plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'protection')
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'protection')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'protection')


# ------ misinformation! -------------
# choose from one of the following dictionaries!
# get all the data from the risk bifurcation data!
eq1_risk_lp1_par_dict = {'risk':  0.12952930209570576, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

eq1_risk_lp1_ss = { 'x1':  0.02179992969509327,
                    'x2':  0.21186033176092678,
                    'x3':  0.07439263859553721,
                    'x4':  0.6784593698650656,
                    'x5':  0.011113221022650485}

eq1_risk_h1_ss = { 'x1':  0.16525533433723974,
                    'x2':  0.4608116685077331,
                    'x3':  0.09068387293929851,
                    'x4':  0.14189412776170351,
                    'x5':  0.0003737491664098206}

eq1_risk_h1_par_dict = {'risk': 1.6352957874550589, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

eq1_risk_bp1_ss = {'x1':  0.02179992969509327,
                'x2':  0.21186033176092678,
                'x3':  0.07439263859553721,
                'x4':  0.6784593698650656,
                'x5':  0.011113221022650485}

eq1_risk_bp1_par_dict = {'risk':  0.12952930209570576, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

# saddle node of the risk bifurcation!
par_dict = eq1_risk_h1_par_dict
ss_dict = eq1_risk_h1_ss

# hopf bifurcation of the risk bifurcation!
#par_dict = eq1_risk_h1_par_dict
#ss_dict = eq1_risk_h1_ss

# branching point bifurcation of the risk bifurcation!
#par_dict = eq1_risk_bp1_par_dict
#ss_dict = eq1_risk_bp1_ss

# compute bifurcation in misinformation!
PC_misinformation = generate_misinformation_bifurcation(ode, ics_dict = ss_dict, par_dict = par_dict, tend = 300, max_points = 120)

# get the data
par_dict, ss_dict, data = get_data(PC_misinformation, curve = 'EQmisinformation', special_point = 'H1', par_dict = par_dict, par = 'misinformation')

# generate a pointset and plot the time around the bifurcation!
pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = 5_000, par = 'misinformation', random_bool = True, eps = 0.05)



# plot a few nulllclines
# sg vs sb nullcline
plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'misinformation')
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'misinformation')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'misinformation')


# -- two dimension bifurcation with respect to misinformation along the x-axis
# generate a bifurcation with respect to the double
ypar_bin = ['education', 'infection_bad', 'infection_good', 'risk', 'recovery', 'protection']
name_curve_bin = ['HO1', 'HO2', 'HO3', 'HO4', 'HO5', 'HO6']
for p, p0 in enumerate(ypar_bin):
    generate_limit_cycle_boundary(PC = PC_misinformation, special_point = 'H1', xpar = 'misinformation', ypar = p0, par_dict = par_dict, max_n_points = 75, curve_type = 'H-C1', name_curve = name_curve_bin[p], xmax = 1, ymax = 1)






# ------- education! -------
# using the dictionaries for the education

# generate a bifurcation!
eq1_h1_ss_new = {'x1': 0.20993799300477495,
                   'x2': 0.48221092545757344,
                   'x3': 0.07073195797121161,
                   'x4': 0.1168184171123022,
                   'x5': 0.00018891967673741587}

eq1_h1_par_dict_new = eq1_risk_bp1_par_dict
eq1_h1_par_dict_new['risk'] = 1.635295791362042

par_dict = eq1_h1_par_dict_new
ss_dict = eq1_h1_ss_new

# -- generating a bifurcation plot with respect to education!
PC_education = generate_education_bifurcation(ode, ics_dict = ss_dict, par_dict = par_dict, tend = 300, max_points = 120)

# get the data
par_dict, ss_dict, data = get_data(PC_education, curve = 'EQeducation', special_point = 'H1', par_dict = par_dict, par = 'education')

# generate a pointset and plot the time around the bifurcation!
pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = 5_000, par = 'education', random_bool = True, eps = 0.05)


# plot a few nulllclines
# sg vs sb nullclines
plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'education')
# sb vs ib nullclines
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'education')
# ib vs v nullclines
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200, par = 'education')



# -- two dimension bifurcation with respect to education along the x-axis
# generate a bifurcation with respect to the double
ypar_bin = ['misinformation', 'infection_bad', 'infection_good', 'risk', 'recovery', 'protection']
name_curve_bin = ['HO11', 'HO21', 'HO31', 'HO41', 'HO51', 'HO61']
for p, p0 in enumerate(ypar_bin):
    generate_limit_cycle_boundary(PC = PC_education, special_point = 'H1', xpar = 'education', ypar = p0, par_dict = par_dict, max_n_points = 75, curve_type = 'H-C1', name_curve = name_curve_bin[p], xmax = 1, ymax = 1)
