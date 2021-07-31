import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from sys_dx import sys_dx

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# get dictionary corresponding to multiple parameters and initial conditions!
# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# initialize initial conditions!
# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

# steady state branching point at risk bifurcation!
ss_bp_r = {'x1': 0.00057159126, 'x2': 0.18949223,
        'x3': 0.19704689, 'x4': 0.60433083,
        'x5': 1}

# parameters for branching point at risk bifurcation!
par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# steady states at hopf for the risk bifurcation!
ss_hopf_r = {'x1': 0.107930, 'x2': 0.345919 ,
             'x3': 0.079105, 'x4': 0.393524,
             'x5': 0.001384}


eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}



eq1_lp1_ss = {'x1': 0.021799678852649853,
              'x2': 0.21186000608052885,
              'x3': 0.07439281159652406,
              'x4': 0.6784598802771113,
              'x5': 0.01111338310412665}


eq1_h2_ss = {'x1': 0.0005386052264346348,
             'x2': 0.18947415697215972,
             'x3': 0.1978608081601073,
             'x4': 0.6035663782158089,
             'x5':  1.064278997774266}

eq1_h2_par_dict = par_bp_r
eq1_h2_par_dict['risk'] = 0.342001918986597

eq1_lp1_par_dict = par_bp_r
eq1_lp1_par_dict['risk'] = 0.1295293020919909

eq1_h1_par_dict = par_bp_r
eq1_h1_par_dict['risk'] = 1.635295791362042

par_hopf_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.387844, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927


def plot_nullclines(option='A', PTS='', par_dict=eq1_h1_par_dict, ss_dict=eq1_h1_ss, n_bin=100, xlow=0, xhigh=1.0,
                        ylow=0, yhigh=1.0, quiv_bool=True, ext_bool=False, ss_dict_2=eq1_lp1_ss,
                        par_dict_2=eq1_lp1_par_dict, evecs_bool=False, evecs=None, ics_dict={}, par = 'risk'):
    """
    function to generate nullclines with a quiver plot field determined using sys_dx function
    :param option: 'A' for SG vs IB, 'B' for SB vs IB, 'C' for IB vs V
    :param PTS: pointset, if '', then generate it
    :param par_dict: dictionary of paramaeters
    :param ss_dict: dicitonary of steay state values
    :param n_bin: number of elements in the np array
    :param xlow: lower bound on x variables
    :param xhigh: upper bound on x variable
    :param ylow: lower bound on y variable
    :param yhigh: upper bound on y variable
    :param quiv_bool: boolean for the quiver plot of the nullcline
    :param ext_bool: boolean to extend the current nullcline to include a second fixed point
    :param ss_dict_2: second dictionary for steady state values
    :param par_dict_2: second dictionary for parameter values
    :param evecs_bool: boolean for the eigenvectors used to determine the trajectory of the plane
    :param evecs: user supplied eigenvectors to determine the flow about the fised point
    :param ics_dict: dictionary containing initial conditions
    :return: plotted nullclines in matplotlib figures
    """
    # unpack steady state values
    x1_ss = ss_dict['x1']
    x2_ss = ss_dict['x2']
    x3_ss = ss_dict['x3']
    x4_ss = ss_dict['x4']
    x5_ss = ss_dict['x5']

    ig_ss = 1 - (x1_ss + x2_ss + x3_ss + x4_ss)

    # unpack parameters
    risk = par_dict['risk']
    protection = par_dict['protection']
    belief = par_dict['belief']
    infection_bad = par_dict['infection_bad']
    infection_good = par_dict['infection_good']
    misinformation = par_dict['misinformation']
    education = par_dict['education']
    recovery = par_dict['recovery']

    """
       H Point found 
       ========================== 
       0 : 
       x1  =  0.20993799300477495
       x2  =  0.48221092545757344
       x3  =  0.07073195797121161
       x4  =  0.1168184171123022
       x5  =  0.00018891967673741587
       risk  =  1.6352957791039242
       """

    if ics_dict == {}:
        x1_0 = 0.20
        x2_0 = 0.50
        x3_0 = 0.001
        x4_0 = 0.14
        x5_0 = 0.0003
        ics = {'x1': x1_0, 'x2': x2_0,
               'x3': x3_0, 'x4': x4_0, 'x5': x5_0}
    else:
        ics = ics_dict

    if PTS == '':
        ode = generate_ode(par_dict=par_dict, ics_dict=ics, tf=10_000)
        pts = ode.compute('nulls_traj').sample(dt=5)
    else:
        pts = PTS
    # pts, ss_dict = generate_pointset(ODE = ode, save_bool=False)
    x1_traj = pts['x1']
    x2_traj = pts['x2']
    x3_traj = pts['x3']
    x4_traj = pts['x4']
    x5_traj = pts['x5']
    t = pts['t']

    # determine the functions of nullclines
    if option == 'A' or option == '':
        # sg vs sb
        a = 0  # eigenvector component in x
        b = 1  # eigenvector component in y

        xlab = r'$S_G$'  # label for x axis
        ylab = r'$S_B$'  # label for y axis

        xnull_lab = r'$N_{S_G}$'  # label for legend
        ynull_lab = r'$N_{S_B}$'

        x_traj = x1_traj
        y_traj = x2_traj

        dx_x = 'x1'  # used for quiver
        dx_y = 'x2'

        x_ss = x1_ss  # steady state value on x
        y_ss = x2_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x2
        x_null = (recovery * ig_ss - x_array * (x5_ss + (infection_good + misinformation) * x3_ss)) / (
                    misinformation * x_array)
        # y null --> solve for x = x1
        y_null = (x3_ss * (infection_bad * y_array - recovery)) / (misinformation * (y_array + x3_ss))

    elif option == 'B':
        # sb vs ib
        a = 1  # eigenvector component in x
        b = 2  # eigenvector component in y

        xlab = r'$S_B$'  # label for x axis
        ylab = r'$I_B$'  # label for y axis

        xnull_lab = r'$N_{S_B}$'  # label for legend
        ynull_lab = r'$N_{I_B}$'

        x_traj = x2_traj
        y_traj = x3_traj

        dx_x = 'x2'  # used for quiver
        dx_y = 'x3'

        x_ss = x2_ss  # steady state value on x
        y_ss = x3_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x3
        x_null = (misinformation * x1_ss * x_array) / (infection_bad * x_array - recovery - misinformation * x1_ss)
        # y null --> solve for x = x2
        y_null = (recovery + education * (x1_ss + ig_ss) - x5_ss * x1_ss / y_array) / infection_bad

    elif option == 'C':
        # ib vs v
        a = 2  # eigenvector component in x
        b = 3  # eigenvector component in y

        xlab = r'$I_B$'  # label for x axis
        ylab = r'$V$'  # label for y axis

        xnull_lab = r'$N_{I_B}$'  # label for legend
        ynull_lab = r'$N_{V}$'

        x_traj = x3_traj
        y_traj = x4_traj

        dx_x = 'x3'  # used for quiver
        dx_y = 'x4'

        x_ss = x3_ss  # steady state value on x
        y_ss = x4_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        # x_null = (misinformation * x2_ss - recovery + education * (x2_ss + x_array - 1)) / ((protection - 1) * infection_good)
        x_null = np.ones(n_bin) * (recovery + education * (x1_ss + ig_ss) - infection_bad * x2_ss) / ((1 - protection) * infection_good)
        # y null --> solve for x = x3
        y_null = (x5_ss * x1_ss) / ((1 - protection) * infection_good * y_array)

    elif option == 'D':
        # ig vs ib
        a = None  # eigenvector component in x
        b = 3  # eigenvector component in y

        xlab = r'$I_G$'  # label for x axis
        ylab = r'$I_B$'  # label for y axis

        xnull_lab = r'$N_{I_G}$'  # label for legend
        ynull_lab = r'$N_{I_B}$'

        x_traj = 1 - (x1_traj + x2_traj + x3_traj)
        y_traj = x4_traj

        dx_x = 'x3'  # used for quiver
        dx_y = 'x4'

        x_ss = x3_ss  # steady state value on x
        y_ss = x4_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        x_null = (misinformation * x2_ss - recovery + education * (x2_ss + x_array - 1)) / (
                (protection - 1) * infection_good)

        # y null --> solve for x = x3
        y_null = (x5_ss * x1_ss) / ((1 - protection) * infection_good * y_array)

    # generate a phase field
    if quiv_bool:
        x, y = np.linspace(xlow, xhigh, 15), np.linspace(ylow, yhigh, 15)
        x1, y1 = np.meshgrid(x, y)
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict=par_dict, ss_dict=ss_dict, xvar=dx_x, yvar=dx_y)
        # normalize growth rate!
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1  # avoid division of zero
        dx1 /= M
        dy1 /= M  # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot='mid')

    z = int(len(x1_traj) / 4)
    plt.plot(x_array, x_null, 'b', label=xnull_lab)
    plt.plot(y_null, y_array, 'r', label=ynull_lab)

    if dx_x == 'x3':
        plt.vlines(x=0, ymin=ylow, ymax=yhigh, color='b')
    if dx_y == 'x3':
        plt.hlines(y=0, xmin=xlow, xmax=xhigh, color='r')

    # plot steady state values!
    plt.plot(x_ss, y_ss, 'ko', fillstyle='left', ms=10)

    # plot eigenvectors, if boolean is true
    if evecs_bool == True:
        evec_real = evecs.real  # real part of the eigenvalue
        # determine the components to use of the eigenvalue depending on the x and y values used for the nullclines
        v1 = np.array([evec_real[a][a], evec_real[b][a]])
        v2 = np.array([evec_real[b][a], evec_real[b][b]])

        # determine the angle between the eigenvectors
        evecs_angle = np.arccos(np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

        # plot the eigenvector's starting from the fied point
        plt.arrow(x_ss, y_ss, v1[0], v1[1], color='b', ls='--')
        plt.arrow(x_ss, y_ss, v2[0], v2[1], color='r', ls='--')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.plot(x_traj[0:], y_traj[0:], 'k')
    plt.legend()
    file_name = f"\{par}_{dx_x}_{dx_y}_nullcline.jpeg"
    plt.savefig(path + file_name, dpi = 300)
    plt.show()

    # time plot
    """plt.plot(t, x1_traj, 'b-', label='$S_G$')
    plt.plot(t, x2_traj, 'r--', label='$S_B$')
    plt.plot(t, x3_traj, 'r', label='$I_B$')
    plt.plot(t, x4_traj, 'b:', label='$V$')
    plt.plot(t, x5_traj, 'm:', label='$\phi$')
    plt.legend()
    plt.xlabel('t (Days)')
    plt.ylabel('Fraction of the Population')
    plt.show()"""