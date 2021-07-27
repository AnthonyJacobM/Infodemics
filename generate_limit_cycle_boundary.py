import PyDSTool as dst
from PyDSTool import *
import numpy as np
from generate_ode import generate_ode
from sys_dx import sys_dx
import matplotlib.pyplot as plt
from generate_bifurcation_2d import generate_bifurcation_2d

path = r'D:\Users\antho\PycharmProjects\Infodemics\figures'


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

def generate_limit_cycle_boundary(PC, curve = 'EQmisinformation', special_point = 'H1', xpar = 'misinformation', ypar = 'education', par_dict = eq2_h2_par_dict, max_n_points = 35, curve_type = 'H-C1', name_curve = 'HO1', xmax = 1, ymax = 1):
    """
    a function to generate the limit cycle boundary for two parameters
    :param PC: Python Continuation used for initializing at a hop-f point!
    :param curve: the curve named for the 1d bifurcation in xpar
    :param special_point: H1, H2, etc. a hopf bifurcation obtained in the previous plots
    :param xpar: parameter along the x axis that will be varied (used as a bifurcation parameter in the PC)
    :param ypar: parameter along the y axis that will be varied
    :param par_dict: dictionary for the parameters used to obtain the Hopf bifurcation!
    :param max_n_points: maximum number of points in the continuation
    :param curve_type: type of method: H-C1 or H-C2
    :param name_curve: name of the curve used for the 2d bifurcation
    :return: plotted figure in matplotlib
    """


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
    PC_2d = generate_bifurcation_2d(PC, special_point=special_point, xpar=xpar,
                                                          ypar=ypar, par_dict = par_dict,
                                                          name_curve=name_curve, max_n_points=70, curve_type=curve_type, curve = curve)
    # get the data obtained in the figure!
    lc_x = PC_2d[name_curve].sol[xpar]
    lc_y = PC_2d[name_curve].sol[ypar]

    # get state variables along the boundary
    x1_ss = PC_2d[name_curve].sol['x1']
    x2_ss = PC_2d[name_curve].sol['x2']
    x3_ss = PC_2d[name_curve].sol['x3']
    x4_ss = PC_2d[name_curve].sol['x4']
    x5_ss = PC_2d[name_curve].sol['x5']

    # fill in between to obtain the limit cycle boundary
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # generate a plot for the limit cycle boundary!
    plt.plot(lc_x, lc_y, 'b', lw = 5)
    plt.fill(lc_x, lc_y, 'k')
    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    #plt.xlim([0, xmax])
    #plt.ylim([0, ymax])

    # get the current axes limits
    xlimit = np.array(ax.get_xlim())
    ylimit = np.array(ax.get_ylim())

    par_x_lb = xlimit[0]
    par_x_ub = xlimit[-1]
    par_y_lb = ylimit[0]
    par_y_ub = ylimit[-1]

    print(par_x_lb, par_x_ub, par_y_lb, par_y_ub)

    xlimit = np.array([par_x_lb, par_x_ub])
    ylimit = np.array([par_y_lb, par_y_ub])

    file_name = f"\lc_{xpar}_{ypar}.jpeg"
    plt.savefig(path + file_name, dpi = 300)
    plt.show()

    # get axes limit
    xlimit = ax.get_xlim()
    ylimit = ax.get_ylim()
    print('xlimit: ', xlimit, 'ylimit: ', ylimit)


    # save the data to a numpy array for later
    np.savetxt(f"lc_{xpar}_{ypar}_xsol.txt", lc_x, delimiter = ',')
    np.savetxt(f"lc_{xpar}_{ypar}_ysol.txt", lc_y, delimiter = ',')
    np.savetxt(f"lc_{xpar}_{ypar}_xlimit.txt", np.array(xlimit), delimiter = ',')
    np.savetxt(f"lc_{xpar}_{ypar}_ylimit.txt", np.array(ylimit), delimiter = ',')

    # save the steady state data!
    np.savetxt(f"lc_{xpar}_{ypar}_x1_ss.txt", x1_ss, delimiter=',')
    np.savetxt(f"lc_{xpar}_{ypar}_x2_ss.txt", x2_ss, delimiter=',')
    np.savetxt(f"lc_{xpar}_{ypar}_x3_ss.txt", x3_ss, delimiter=',')
    np.savetxt(f"lc_{xpar}_{ypar}_x4_ss.txt", x4_ss, delimiter=',')
    np.savetxt(f"lc_{xpar}_{ypar}_x5_ss.txt", x5_ss, delimiter=',')
