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
from generate_heat_map_limit_cycle import generate_heat_map_limit_cycle
from plot_limit_cycle_heatmap import plot_limit_cycle_heatmap
from plot_lc_heatmap_revised import plot_lc_heatmap


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

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

# ---- risk
PC_risk, par_dict_risk, ss_dict_risk, data_risk = plot_risk_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H1', tend = 10_000, eps0 = 0.00001)


aux_par_list = ['misinformation', 'education', 'recovery', 'infection_bad']
max_step_size_bin = [90, 75, 85, 65, 65]
x_lb_bin = [1.2, 1.2, 1.2, 1.2]
x_ub_bin = [2.0, 2.0, 2.0, 2.0]
y_lb_bin = [0.04, 0, 0.05, 0.1]
y_ub_bin = [0.16, 0.8, 0.14, 0.85]

aux_par_list = ['misinformation', 'education', 'recovery', 'infection_bad']
max_step_size_bin = [80, 155, 65, 65]
x_lb_bin = [1.3, 1.3, 1.3, 1.3]
x_ub_bin = [2.0, 2.0, 2.0, 2.0]
y_lb_bin = [0, 0, 0.05, 0.1]
y_ub_bin = [0.2, 0.8, 0.14, 0.85]

#aux_par_list = ['education']
#max_step_size_bin = [90]
#y_lb_bin = [0]
#y_ub_bin = [0.5]


# generate limit cycle boundaries for each auxillary parameter
for p, p0 in enumerate(aux_par_list):
    generate_limit_cycle_boundary(PC_risk, curve = 'EQrisk',
                                  special_point = 'H1', xpar = 'risk',
                                  ypar = p0, par_dict = eq1_h1_par_dict,
                                  max_n_points = max_step_size_bin[p],
                                  curve_type = 'H-C1', name_curve = 'Hrisk' + str(p))

    """generate_heat_map_limit_cycle(PC_risk, curve='EQrisk',
                              special_point='H1', xpar='risk',
                              ypar=p0, par_dict=par_dict_risk,
                              ss_dict=ss_dict_risk, max_n_points=max_step_size_bin[p],
                              curve_type='H-C1',  name_curve = 'Hrisk_new' + str(p), n_bin=100, tend=5_000, eps=0.0005)"""

    file_vac = f"risk_{p0}_vac.txt"
    file_bad = f"risk_{p0}_bad.txt"
    file_inf = f"risk_{p0}_inf.txt"
    file_xlimit = f"lc_risk_{p0}_xlimit.txt"
    file_ylimit = f"lc_risk_{p0}_ylimit.txt"


    data_vac = np.loadtxt(file_vac, delimiter = ',', dtype = float)
    data_bad = np.loadtxt(file_bad, delimiter = ',', dtype = float)
    data_inf = np.loadtxt(file_inf, delimiter = ',', dtype = float)
    #xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
    #ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)
    
    plot_lc_heatmap(data = data_vac, zvar = 'inf', xpar='risk', ypar = p0, cmap = 'Reds', xlow=x_lb_bin[p], xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_lc_heatmap(data = data_vac, zvar = 'bad', xpar='risk', ypar = p0, cmap = 'Greys', xlow=x_lb_bin[p], xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_lc_heatmap(data = data_vac, zvar = 'vac', xpar='risk', ypar = p0, cmap = 'Blues', xlow=x_lb_bin[p], xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])


quit()

# --- protection
#PC_protection, par_dict_protection, ss_dict_protection, data_protection = plot_protection_bifurcation()

# --- misinformation
PC_misinformation, par_dict_misinformation, ss_dict_misinformation, data_misinformation = plot_misinformation_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.005)

aux_par_list = ['protection', 'risk', 'education', 'recovery', 'infection_good', 'infection_bad']
max_step_size_bin = [90, 100, 100, 85, 65, 65, 65]
x_lb_bin = [0, 0, 0, 0, 0, 0.05]
x_ub_bin = [0.5, 0.25, 0.14, 0.50, 0.20, 0.20]
y_lb_bin = [0.2, 0.9, 0, 0, 0, 0.10]
y_ub_bin = [1.0, 1.7, 0.40, 0.08, 0.5, 0.55]

"""aux_par_list = ['infection_bad']
x_lb_bin = [0.05]
x_ub_bin = [0.2]
y_lb_bin = [0.10]
y_ub_bin = [0.55]
"""
# generate limit cycle boundaries for each auxillary parameter
for p, p0 in enumerate(aux_par_list):
    """generate_limit_cycle_boundary(PC_misinformation, curve = 'EQmisinformation',
                                  special_point = 'H1', xpar = 'misinformation',
                                  ypar = p0, par_dict = par_dict_risk,
                                  max_n_points = max_step_size_bin[p],
                                  curve_type = 'H-C1', name_curve = 'Hmisinformation' + str(p))"""
    generate_heat_map_limit_cycle(PC_misinformation, curve='EQmisinformation',
                                  special_point='H1', xpar='misinformation',
                                  ypar=p0, par_dict=par_dict_misinformation,
                                  ss_dict=ss_dict_misinformation, max_n_points=max_step_size_bin[p],
                                  curve_type='H-C1', name_curve='Hmisinfo_new' + str(p), n_bin=100, tend=7_500, eps=0.001)

    file_vac = f"misinformation_{p0}_vac.txt"
    file_bad = f"misinformation_{p0}_bad.txt"
    file_inf = f"misinformation_{p0}_inf.txt"
    file_xlimit = f"lc_misinformation_{p0}_xlimit.txt"
    file_ylimit = f"lc_misinformation_{p0}_ylimit.txt"

    data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
    data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
    data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)
    #xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
    #ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

    plot_limit_cycle_heatmap(data=data_vac, zvar='inf', xpar='misinformation', ypar=p0, cmap='Reds', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_limit_cycle_heatmap(data=data_vac, zvar='bad', xpar='misinformation', ypar=p0, cmap='Greys', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_limit_cycle_heatmap(data=data_vac, zvar='vac', xpar='misinformation', ypar=p0, cmap='Blues', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])

quit()

# --- education
PC_education, par_dict_education, ss_dict_education, data_education = plot_education_bifurcation(par_dict = par_dict_misinformation,
                                                                                                 ics_dict = ss_dict_misinformation,
                                                                                                 special_point = 'H1', tend = 5_000, eps0 = 0.001)


aux_par_list = ['risk', 'education', 'recovery', 'infection_good', 'infection_bad', 'protection']
max_step_size_bin = [100, 100, 85, 65, 65, 65, 80]

# generate limit cycle boundaries for each auxillary parameter
for p, p0 in enumerate(aux_par_list):
    """generate_limit_cycle_boundary(PC_education, curve = 'EQeducation',
                                  special_point = 'H1', xpar = 'education',
                                  ypar = p0, par_dict = par_dict_risk,
                                  max_n_points = max_step_size_bin[p],
                                  curve_type = 'H-C1', name_curve = 'Heducation' + str(p))"""
    generate_heat_map_limit_cycle(PC_education, curve='EQeducation',
                                  special_point='H1', xpar='education',
                                  ypar=p0, par_dict=par_dict_education,
                                  ss_dict=ss_dict_education, max_n_points=max_step_size_bin[p],
                                  curve_type='H-C1', name_curve='Heducation_new' + str(p), n_bin=100, tend=7_500, eps=0.001)

    file_vac = f"education_{p0}_vac.txt"
    file_bad = f"education_{p0}_bad.txt"
    file_inf = f"education_{p0}_inf.txt"
    file_xlimit = f"lc_education_{p0}_xlimit.txt"
    file_ylimit = f"lc_education_{p0}_ylimit.txt"

    data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
    data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
    data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)
    # xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
    # ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

    plot_limit_cycle_heatmap(data=data_inf, zvar='inf', xpar='education', ypar=p0, cmap='Reds', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_limit_cycle_heatmap(data=data_bad, zvar='bad', xpar='education', ypar=p0, cmap='Greys', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])
    plot_limit_cycle_heatmap(data=data_vac, zvar='vac', xpar='education', ypar=p0, cmap='Blues', xlow=x_lb_bin[p],
                             xhigh=x_ub_bin[p], ylow=y_lb_bin[p], yhigh=y_ub_bin[p])



"""
# --- recovery
PC_reocvery, par_dict_recovery, ss_dict_recovery, data_recovery = plot_recovery_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'H2', tend = 5_000, eps0 = 0.15)



# --- infection_good
PC_infection_good, par_dict_infection_good, ss_dict_infection_good, data_infection_good = plot_infection_good_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'LP1', tend = 5_000, eps0 = 0.15)



# --- infection_bad
PC_infection_bad, par_dict_infection_bad, ss_dict_infection_bad, data_infection_bad = plot_infection_bad_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss,
                                special_point = 'LP1', tend = 5_000, eps0 = 0.15)

"""