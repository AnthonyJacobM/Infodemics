import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PyDSTool as dst
from PyDSTool import *
from gen_sys import gen_sys
from generate_ode import generate_ode
from perturb_ics import perturb_ics
from plot_nullclines import plot_nullclines
from plot_risk_bifurcation import plot_risk_bifurcation
from plot_education_bifurcation import plot_education_bifurcation
from plot_misinformation_bifurcation import plot_misinformation_bifurcation

# begin
# first, generate an ode
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
                'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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


# Generate new bifurcations
#PC_risk, par_dict, ss_dict, data = plot_risk_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, special_point = 'H1', tend = 5_000, eps0 = 0.15)
#ode = generate_ode(par_dict = par_dict, ics_dict = ss_dict, tf = 100)

PC_education, par_dict, ss_dict, data = plot_education_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, special_point = 'H1', tend = 5_000, eps0 = 0.15)
ode = generate_ode(par_dict = par_dict, ics_dict = ss_dict, tf = 100)
print('dicitonary of parameter: ', par_dict)
#PC_misinformation, par_dict, ss_dict, data = plot_misinformation_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, special_point = 'H1', tend = 5_000, eps0 = 0.15)
#ode = generate_ode(par_dict = par_dict, ics_dict = ss_dict, tf = 100)


# generate new nullcline figures
def plot_nullclines_sequence(ode, option = 'B', PC = PC_education, curve = 'EQrisk', par_dictionary = par_dict, ss_dictionary = ss_dict, data_new = data, par = 'risk'):
    """
    function to plot a collection of nullclines along the bifurcation plot
    :param ode: ode used for simulation
    :param PC: python continuation
    :param curve: curve along the continuation
    :param par_dictionary: dictionary of parameters
    :param ss_dictionary: dictionary of steady state values
    :param data_new: data obatined for the bifurcations
    :param par: parameter string used for the bifurcation plot
    :return: save figures and plotted nullclines
    """
    # begin
    # get labels of the data
    if par == 'risk':
        p_label = r'$r$'
    elif par == 'protection':
        p_label = r'$\delta$'
    elif par == 'education':
        p_label = r'$\epsilon$'
    elif par == 'misinformation':
        p_label = r'$\mu$'
    elif par == 'infection_good':
        p_label = r'$\chi$'
    elif par == 'infection_bad':
        p_label = r'$\hat{\chi}$'
    elif par == 'recovery':
        p_label = r'$\gamma$'
    else:
        p_label = ''
        print('choose one of the default parameters')
        quit()

    # now we get the data
    x1_sol = PC[curve].sol['x1']
    x2_sol = PC[curve].sol['x2']
    x3_sol = PC[curve].sol['x3']
    x4_sol = PC[curve].sol['x4']
    x5_sol = PC[curve].sol['x5']
    par_sol = PC[curve].sol[par]
    ss_dict = ics_dict_def

    # now we iterate over entry and plot the nullclines
    for p, p0 in enumerate(par_sol):
        # change p0 number
        p0 = np.round(p0, 3)
        par_dict[par] = p0 # change the parameter value
        # change the steady states
        ss_dict['x1'] = x1_sol[p]
        ss_dict['x2'] = x2_sol[p]
        ss_dict['x3'] = x3_sol[p]
        ss_dict['x4'] = x4_sol[p]
        ss_dict['x5'] = x5_sol[p]

        # flag for WHILE loop
        FLAG = 0
        # generate a perturbation about the steady state
        while FLAG == 0:
            ss_new = perturb_ics(ics_dict = ss_dict, eps0 = 0.05)
            #ode.set(ics = ss_new, pars = par_dict)
            # generate a pointset
            z, t = gen_sys(par_dict = par_dict, ics_dict = ss_new)
            sg, sb, ib, v, phi = z.T
            x1_max = np.max(sg)
            x1_min = np.min(sg)
            x2_max = np.max(sb)
            x2_min = np.min(sb)
            x3_max = np.max(ib)
            x3_min = np.min(ib)
            x4_max = np.max(v)
            x4_min = np.min(v)
            x5_max = np.max(phi)
            x5_min = np.min(phi)

            pts = {'x1': sg, 'x2': sb, 'x3': ib, 'x4': v, 'x5': phi, 't': t}

            #print(x1_max, x2_max, x3_max, x4_max, x5_max, x1_min, x2_min, x3_min, x4_min, x5_min)

            if x1_max <= 1 and x2_max <= 1 and x3_max <= 1 and x4_max <= 1 and x5_max <= 1 and x1_min >= 0 and x2_min >= 0 and x3_min >= 0 and x4_min >= 0 and x5_min >= 0:
                FLAG = 1




        title = p_label + ' = ' + str(p0)
        plot_nullclines(option = option, par_dict = par_dict, ss_dict = ss_dict, PTS = pts,
                        w0 = 0, z0 = 0, distance = 0.05, seq = str(p), title = title, title_bool = True)


#plot_nullclines_sequence(ode = ode, option = 'A', par = 'education', curve = 'EQeducation', PC = PC_education)
plot_nullclines_sequence(ode = ode, option = 'B', par = 'education', curve = 'EQeducation', PC = PC_education)
#plot_nullclines_sequence(ode = ode, option = 'C', par = 'education', curve = 'EQeducation', PC = PC_education)