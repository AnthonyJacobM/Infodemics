import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from generate_a_bifurcation import generate_a_bifurcation
from plot_nullclines import plot_nullclines
from get_data import get_data
from gen_sys import gen_sys
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
            'risk': 1.5, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
"""ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}"""

ics_dict_def = {'x1': 0.25, 'x2': 0.25,
                'x3': 0.25, 'x4': 0.0, 'x5': 0.5}

# dictionaries for parameters obtained on the LP1 previously!
eq1_risk_lp1_par_dict = {'risk': 0.12952930209570576, 'protection': 0.90,
                         'recovery': 0.07, 'belief': 1.0,
                         'education': 0.33, 'misinformation': 0.10,
                         'infection_good': 0.048, 'infection_bad': 0.37,
                         'ace': 0}

eq1_risk_lp1_ss = {'x1': 0.02179992969509327,
                   'x2': 0.21186033176092678,
                   'x3': 0.07439263859553721,
                   'x4': 0.6784593698650656,
                   'x5': 0.011113221022650485}

eq1_risk_bp1_ss = {'x1': 0.000571591255633369,
                   'x2': 0.18949223460848785,
                   'x3': 0.19704689218431548,
                   'x4': 0.6043308284146786,
                   'x5': 1.0}
eq1_risk_bp1_par_dict = par_dict_def
eq1_risk_bp1_par_dict['risk'] = 0.3402198531896148

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042


def plot_a_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, special_point = 'H1', tend = 5_000, eps0 = 0.15):
    """
    function to plot codim-1 bifurcation, nullclines, and perturbed time evolution
    :param par_dict: dictionary of parameters used to simulate the bifurcation
    :param ics_dict: dictionary of initial conditions used to simulate the bifurcation
    :param special_point: point to grab the data on the codim-1 bifurcation
    :param tend: final time used for pointset
    :param eps0: value to perturb the steady states as ics in plotting time
    :return: PC_risk the continuation curve that contains all the data
    """
    # generate the ode
    ode = generate_ode(par_dict, ics_dict = ics_dict, tf=500)

    # generate a pointset!
    pts, ss_dict = generate_pointset(ode, save_bool = True)


    # ------------- a -------------------------
    PC_a = generate_a_bifurcation(ode, ics_dict = ics_dict, par_dict = par_dict, tend = 300)

    # get the data
    par_dict, ss_dict, data = get_data(PC_a, curve = 'EQace', special_point = special_point, par_dict = par_dict, par = 'ace')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = tend, par = 'ace', random_bool = True, eps = eps0)


    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.25, yhigh = 0.6, n_bin = 200, par = 'ace')
    # sb vs ib nullcline
    plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.50, yhigh = 0.5, n_bin = 200, par = 'ace')
    # ib vs v nullcline
    plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.25, yhigh = 0.75, n_bin = 200, par = 'ace')
    # ib vs v nullcline
    plot_nullclines(option='E', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=1.0, yhigh=0.75,
                    n_bin=200, par='ace')

    return PC_a, par_dict, ss_dict, data


def test_run(par_dict = par_dict_def, ics_dict = ics_dict_def):
    """
    function to generate the test run of the bifurcation
    :return: plotted bifurcation in "a"
    """
    # generate an ode
    par_dict['ace'] = 0.0
    ode = generate_ode(par_dict = par_dict, ics_dict = ics_dict, tf = 10_000)

    # run a time series to see if the final solution is in a steady states
    z, t = gen_sys(par_dict=par_dict, ics_dict=ics_dict, tf = 10_000)
    sg, sb, ib, v, phi = z.T
    ig = 1 - (sb + sg + v + ib)

    # begin plotting each of the state variables!
    plt.plot(t, sg, 'b--')
    plt.plot(t, sb, 'r--')
    plt.plot(t, ib, 'r')
    plt.plot(t, v, 'k--')
    plt.plot(t, phi, 'k')
    plt.plot(t, ig, 'b')
    plt.show()

    # how does it depend on the information?

    plt.plot(ig[2000:], ib[2000:], 'r')
    plt.show()
    plt.plot(sg[2000:], sb[2000:], 'b')
    plt.show()
    plt.plot(sg[2000:], ib[2000:], 'k')
    plt.show()
    plt.plot(sb[2000:], ig[2000:], 'k')
    plt.show()

    pts = {'t': t, 'x1': sg, 'x2': sb,
           'x3': ib, 'x4': v, 'x5': phi}

    ss_dict = {'x1': np.average(sg[2000:]), 'x2': np.average(sb[2000:]),
           'x3': np.average(ib[2000:]), 'x4': np.average(v[2000:]), 'x5': np.average(phi[2000:])}

    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xlow = None, ylow = None, xhigh=None, yhigh=None,
                    n_bin=200, par='ace')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xlow = None, ylow = None, xhigh=None, yhigh=None,
                    n_bin=200, par='ace')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xlow = None, ylow = None, xhigh=None, yhigh=None,
                    n_bin=200, par='ace')



#test_run()

