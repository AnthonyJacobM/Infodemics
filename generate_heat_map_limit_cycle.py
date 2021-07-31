import PyDSTool as dst
from PyDSTool import *
import numpy as np
from generate_ode import generate_ode
from sys_dx import sys_dx
import matplotlib.pyplot as plt
from generate_bifurcation_2d import generate_bifurcation_2d
from generate_limit_cycle_boundary import generate_limit_cycle_boundary
from scipy.signal import argrelextrema
from plot_heatmaps import plot_heatmaps
from gen_sys import gen_sys

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# initialize as the parameter dictionary for the bifurcation!
# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}


eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

eq2_h2_par_dict = eq1_h1_par_dict
eq2_h2_par_dict['misinformation'] = 0.0660611192767927

def generate_heat_map_limit_cycle(PC, curve = 'EQmisinformation', special_point = 'H1', xpar = 'misinformation', ypar = 'education', par_dict = eq2_h2_par_dict, ss_dict = {}, max_n_points = 35, curve_type = 'H-C1', name_curve = 'HO1', n_bin = 10, tend = 4_000, eps = 0.01, load_bool = False):
    """
    a function to generate the limit cycle boundary for two parameters
    :param PC: Python Continuation used for initializing at a hop-f point!
    :param curve: the curve named for the 1d bifurcation in xpar
    :param special_point: H1, H2, etc. a hopf bifurcation obtained in the previous plots
    :param xpar: parameter along the x axis that will be varied (used as a bifurcation parameter in the PC)
    :param ypar: parameter along the y axis that will be varied
    :param par_dict: dictionary for the parameters used to obtain the Hopf bifurcation!
    :param ss_dict: dictionary for steady state values obtained at the initializer
    :param max_n_points: maximum number of points in the continuation
    :param curve_type: type of method: H-C1 or H-C2
    :param name_curve: name of the curve used for the 2d bifurcation
    :param tend: final time for numerical simulation
    :return: plotted figure in matplotlib
    """
    # generate a figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if xpar == 'risk':
        xlab = '$r$'
    elif xpar == 'education':
        xlab = r'$\tilde{\chi}_{gb}$'
    elif xpar == 'misinformation':
        xlab = r'$\tilde{\chi}_{bg}$'
    elif xpar == 'infection_good':
        xlab = r'$\chi_{gb}$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}_{bb}$'
    elif xpar == 'protection':
        xlab = '$\delta$'
    elif xpar == 'recovery':
        xlab = r'$\gamma$'
    else:
        xlab = ''
        print('choose par from of the following:')
        print('\t risk')
        print('\t protection')
        print('\t education')
        print('\t misinformation')
        print('\t infection_good')
        print('\t infection_bad')
        quit()

    if ypar == 'risk':
        ylab = '$r$'
    elif ypar == 'education':
        ylab = r'$\tilde{\chi}_{gb}$'
    elif ypar == 'misinformation':
        ylab = r'$\tilde{\chi}_{bg}$'
    elif ypar == 'infection_good':
        ylab = r'$\chi_{gb}$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}_{bb}$'
    elif ypar == 'protection':
        ylab = '$\delta$'
    elif ypar == 'recovery':
        ylab = r'$\gamma$'
    else:
        ylab_ext = ''
        print('choose par from of the following:')
        print('\t risk')
        print('\t protection')
        print('\t education')
        print('\t misinformation')
        print('\t infection_good')
        print('\t infection_bad')
        quit()

    # generate a continuation!
    if load_bool == False:
        PC_2d = generate_bifurcation_2d(PC, special_point=special_point, xpar=xpar,                                                          ypar=ypar, par_dict = par_dict,
                                                          name_curve=name_curve, max_n_points=max_n_points, curve_type=curve_type, curve = curve)

        # get the data obtained in the figure!
        lc_x = PC_2d[name_curve].sol[xpar]
        lc_y = PC_2d[name_curve].sol[ypar]

        # get state variables along the boundary
        x1_ss = PC_2d[name_curve].sol['x1']
        x2_ss = PC_2d[name_curve].sol['x2']
        x3_ss = PC_2d[name_curve].sol['x3']
        x4_ss = PC_2d[name_curve].sol['x4']
        x5_ss = PC_2d[name_curve].sol['x5']
    else:
        file_x = f"lc_{xpar}_{ypar}_xsol.txt"
        file_y = f"lc_{xpar}_{ypar}_ysol.txt"
        lc_x = np.loadtxt(file_x, delimiter = ',', dtype = float)
        lc_y = np.loadtxt(file_y, delimiter = ',', dtype = float)

    # get parameter bounds for limit cycle boundary
    lc_x_lb = np.min(lc_x)
    lc_x_ub = np.max(lc_x)
    lc_y_lb = np.min(lc_y)
    lc_y_ub = np.max(lc_y)



    # fill in between to obtain the limit cycle boundary
    # generate a plot for the limit cycle boundary!
    plt.plot(lc_x, lc_y, 'b', lw = 5)
    plt.fill(lc_x, lc_y, 'k')
    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)


    # get current axes limits
    xlimit = ax.get_xlim()
    ylimit = ax.get_ylim()

    # plot the boundary!
    file_name = f"\Boundary_lc_{xpar}_{ypar}.jpeg"
    plt.savefig(path + file_name, dpi = 300)
    plt.show()


    # obtain the lower bound and upper bound in each parameter
    par_x_lb = xlimit[0]
    par_x_ub = xlimit[-1]
    par_y_lb = ylimit[0]
    par_y_ub = ylimit[-1]
    # generate parameter array's for to obtain the limit
    par_x_bin = np.linspace(par_x_lb, par_x_ub, n_bin)
    par_y_bin = np.linspace(par_y_lb, par_y_ub, n_bin)
    # generate arrays for infected, vaccinated and bad
    inf_bin = np.zeros((n_bin, n_bin))
    vac_bin = np.zeros((n_bin, n_bin))
    bad_bin = np.zeros((n_bin, n_bin))





    # loop through parameter x and parameter y bin's and get the peak or steady state values
    for x, x0 in enumerate(par_x_bin):
        for y, y0 in enumerate(par_y_bin):
            print(x, y, 'out of', n_bin, n_bin)
            # change parameters in x and y
            par_dict[xpar] = x0
            par_dict[ypar] = y0

            inf_peak = 1.25  # initialized to go inside the loop
            vac_peak = 1.25
            bad_peak = 1.25
            # generate a while loop when the peaks are bounded by 1
            while vac_peak > 1 or inf_peak > 1 or bad_peak > 1:
                ss_dict_perturbed = {}
                eps0 = eps

                # generate new initial conditions!
                for k, v in ss_dict.items():
                    sgn = np.sign(np.random.rand(1) - 1 / 2)[0]
                    ss_dict_perturbed[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1))[0]  # perturb new steady state

                """# generate new system
                ode = generate_ode(par_dict, ss_dict_perturbed, tf=tend)  # generate a pointset
                pts = ode.compute('sample').sample(dt=1)
                # get state variables
                sg = pts['x1']
                sb = pts['x2']
                ib = pts['x3']
                v = pts['x4']"""


                z, t = gen_sys(par_dict = par_dict, ics_dict = ss_dict_perturbed, tf = tend)
                sg, sb, ib, v, phi = z.T

                inf = 1 - (sg + sb + v) # get infected
                bad = sb + ib # get bad


                # get all the peaks of infected and bad
                peak_inf_idx = argrelextrema(inf, np.greater)[0]
                peak_bad_idx = argrelextrema(bad, np.greater)[0]
                peak_vac_idx = argrelextrema(v, np.greater)[0]


                # get the maximum of all the peaks
                if len(peak_inf_idx) > 0:
                    inf_peak = np.average(inf[peak_inf_idx])
                else:
                    inf_peak = inf[-1]
                if len(peak_bad_idx) > 0:
                    bad_peak = np.average(bad[peak_bad_idx])
                else:
                    bad_peak = bad[-1]
                if len(peak_vac_idx) > 0:
                    vac_peak = np.average(v[peak_vac_idx])
                else:
                    vac_peak = v[-1]

                # if they are all bounded continue
                if vac_peak <= 1 and inf_peak <= 1 and bad_peak <= 1:
                    break
                else:
                    continue

            # append to the data!
            inf_bin[x][y] = inf_peak
            bad_bin[x][y] = bad_peak
            vac_bin[x][y] = vac_peak
            print('inf: ', inf_peak, 'vac: ', vac_peak, 'bad: ', bad_peak)

    # save the data!
    np.savetxt(f"{xpar}_{ypar}_inf.txt", inf_bin, delimiter = ',')
    np.savetxt(f"{xpar}_{ypar}_bad.txt", bad_bin, delimiter = ',')
    np.savetxt(f"{xpar}_{ypar}_vac.txt", vac_bin, delimiter = ',')

    # save the limit cycle boundaries!
    # save the data to a numpy array for later
    np.savetxt(f"lc_{xpar}_{ypar}_xsol.txt", lc_x, delimiter=',')
    np.savetxt(f"lc_{xpar}_{ypar}_ysol.txt", lc_y, delimiter=',')
    #np.savetxt(f"lc_{xpar}_{ypar}_xlimit.txt", np.array(xlimit), delimiter=',')
    #np.savetxt(f"lc_{xpar}_{ypar}_ylimit.txt", np.array(ylimit), delimiter=',')

array = np.random.rand(5)
np.savetxt('test.txt', array, delimiter = ',')
