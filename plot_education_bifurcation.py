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

# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}

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


par_dict_default = eq1_h1_par_dict_new
ss_dict_default = eq1_h1_ss_new

# generate a new bifurcation
eq_mu_h1_ss = {'x1': 0.20993797903197217,
               'x2': 0.48221091917926023,
               'x3': 0.07073196386807669,
               'x4': 0.11681842442668024,
               'x5': 0.00018891971689025862}

eq_mu_h1_par_dict = eq1_h1_par_dict

eq_mu_h1_par_dict['misinformation']  =  0.06606112781121165


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
    PC_education = generate_education_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300, max_points = 1000)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_education, curve='EQeducation', special_point=special_point, par_dict=par_dict,
                                           par='education')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='education', random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, w0 = 0,
                    n_bin=200, par='education', distance = 0.05)

    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, w0 = 0,
                    n_bin=200, par='education', distance = 0.05)

    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, w0 = 0,
                    n_bin=200, par='education', distance = 0.05)

    return PC_education, par_dict, ss_dict, data




def sample(eigs_bool = False):
    """
    test function to plot bifurcations
    :param eigs_bool: boolean to plot the eigenvalues
    :return: plotted bifurcations
    """
    PC, par_dict, ss_dict, data = plot_education_bifurcation(par_dict=eq_mu_h1_par_dict, ics_dict=eq_mu_h1_ss,
                                                             special_point='H3', tend=20_000, eps0=0.05)

    # plotting the bifurcation
    if eigs_bool == True:
        risk_bin = PC['EQprotection'].sol['protection']
        evals_risk = [PC['EQprotection'].sol[z].labels['EP']['data'].evals for z in range(len(PC['EQprotection'].sol))]
        evecs_risk = [PC['EQprotection'].sol[z].labels['EP']['data'].evecs for z in range(len(PC['EQprotection'].sol))]
        stab_risk = [PC['EQprotection'].sol[z].labels['EP']['stab'] for z in range(len(PC['EQprotection'].sol))]
        par_risk = PC['EQprotection'].sol['protection']
        evals_risk_max = [np.max(np.real(evals_risk[z][:])) for z in range(len(evals_risk))]
        evals_risk_idx = [np.argmax(np.real(evals_risk[z][:])) for z in range(len(evals_risk))]
        evals_risk_imag = np.imag(evals_risk)

        plt.plot(risk_bin, evals_risk_max, 'k')
        plt.xlabel('r', fontsize=18)
        plt.ylabel(r'$\mathbb{R}(\lambda)$', fontsize=18)
        plt.show()

        plt.plot(risk_bin, np.max(evals_risk_imag), 'b')
        plt.plot(risk_bin, np.min(evals_risk_imag), 'b')
        plt.xlabel('r', fontsize=18)
        plt.ylabel(r'$\mathbb{C}(\lambda)$', fontsize=18)
        plt.show()

    PC, par_dict, ss_dict, data = plot_education_bifurcation(par_dict=eq_mu_h1_par_dict, ics_dict=eq_mu_h1_ss,
                                                             special_point='H2', tend=20_000, eps0=0.05)

    PC, par_dict, ss_dict, data = plot_education_bifurcation(par_dict=eq_mu_h1_par_dict, ics_dict=eq_mu_h1_ss,
                                                             special_point='LP1', tend=20_000, eps0=0.05)



#sample()