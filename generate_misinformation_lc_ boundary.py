from generate_bifurcation_2d import generate_bifurcation_2d
from generate_risk_bifurcation import generate_risk_bifurcation
from get_data import get_data
from generate_heat_map_limit_cycle import generate_heat_map_limit_cycle
from plot_lc_heatmap_revised import plot_lc_heatmap
from generate_ode import generate_ode
from plot_misinformation_bifurcation import plot_misinformation_bifurcation
from generate_limit_cycle_boundary import generate_limit_cycle_boundary
import matplotlib.pyplot as plt
import numpy as np


# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

# steady state in hopf bifurcation
eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

# dictionary of parameters
eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

def generate_misinformation_lc_boundary(execution = 'generate_lc'):
    """
    :param execution: use generate_lc, generate_hm, plot_hm
    function to geenrate the risk boundary of the limit cycles
    :return: plotted matplotlib fiugre
    """
    # generate a risk bifurcation using the user supplied ics and dictionary of parameters
    PC_misinformation, par_dict_misinformation, ss_dict_misinformation, data_misinformation = plot_misinformation_bifurcation(par_dict=eq1_h1_par_dict,
                                                                            ics_dict=eq1_h1_ss,
                                                                           special_point='H1', tend=0,
                                                                            eps0=0.00001)
    # use auxillary parameters
    aux_par_list = ['education', 'recovery', 'risk']
    max_step_size_bin = [100, 115, 115]
    x_lb_bin = [1.2, 1.2, 1.2, 1.2]
    x_ub_bin = [2.0, 2.0, 2.0, 2.0]
    y_lb_bin = [0.04, 0, 0.05, 0.1]
    y_ub_bin = [0.16, 0.8, 0.14, 0.85]

    aux_par_list = ['education', 'risk', 'protection', 'recovery', 'infection_good', 'infection_bad']
    max_step_size_bin = [75, 75, 100, 85, 65, 65, 65]

    aux_par_list = ['protection']
    max_step_size_bin = [85]

    # iterate over the array and determine the limit cycle boundary
    for p, p0 in enumerate(aux_par_list):
        if execution == 'generate_lc':
            generate_limit_cycle_boundary(PC_misinformation, curve='EQmisinformation',
                                          special_point='H1', xpar='misinformation',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          max_n_points=max_step_size_bin[p],
                                          curve_type='H-C1', name_curve='Hmisinformation' + str(p),
                                          save_bool=True)
        elif execution == 'generate_hm':
            generate_heat_map_limit_cycle(PC_misinformation, curve='EQmisinformation',
                                          special_point='H1', xpar='misinformation',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          ss_dict=eq1_h1_ss, max_n_points=max_step_size_bin[p], load_bool = True,
                                          curve_type='H-C1',  name_curve = 'Hmisinformation_new' + str(p), n_bin=100, tend=5_000, eps=0.0005)
        else:
            file_vac = f"misinformation_{p0}_vac.txt"
            file_bad = f"misinformation_{p0}_bad.txt"
            file_inf = f"misinformation_{p0}_inf.txt"
            file_xlimit = f"lc_misinformation_{p0}_xlimit.txt"
            file_ylimit = f"lc_misinformation_{p0}_ylimit.txt"

            data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
            data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
            data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)
            xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
            ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

            plot_lc_heatmap(data=data_vac, zvar='inf', xpar='misinformation', ypar=p0, cmap='Reds',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_vac, zvar='bad', xpar='misinformation', ypar=p0, cmap='Greys',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_vac, zvar='vac', xpar='misinformation', ypar=p0, cmap='Blues',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])

#generate_misinformation_lc_boundary(execution = 'generate_lc')
#generate_misinformation_lc_boundary(execution = 'generate_hm')
generate_misinformation_lc_boundary(execution = 'plot_hm')


