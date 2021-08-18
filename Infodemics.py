#!/usr/bin/env python
# coding: utf-8

# In[269]:


import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PyDSTool.Toolbox import phaseplane as pp
from scipy.signal import argrelextrema

# plotting arguments for bifurcation: go here:
# https://pydstool.github.io/PyDSTool/PyCont.html#head-02adc8e1f50100e61e228102f288fdf98c957b2d

# PyCont documentation for reference
# https://pydstool.github.io/PyDSTool

# change defaults of matplotlx3 to publication fonts and linewidths
# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

# initialize base parameters

# recovery = \gamma, belief = m, risk = r, protection = \delta, education = \tilde{\chi}_{gb}, misinformation = \tilde{\chi}_{bg}
# infection_good = \chi_{bg}, infection_bad = \chi_{bb}

par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# initialize initial conditions
# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

ss_bp_r = {'x1': 0.00057159126, 'x2': 0.18949223,
        'x3': 0.19704689, 'x4': 0.60433083,
        'x5': 1}

par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

ss_hopf_r = {'x1': 0.107930, 'x2': 0.345919 ,
             'x3': 0.079105, 'x4': 0.393524,
             'x5': 0.001384}

eq1_lp1_ss = {'x1': 0.021799678852649853,
              'x2': 0.21186000608052885,
              'x3': 0.07439281159652406,
              'x4': 0.6784598802771113,
              'x5': 0.01111338310412665}

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}



# for risk bifurcation
evec_eq1_h1 = np.array([[-0.90954552+0.j,         -0.09156548+0.41631257j, -0.09156548-0.41631257j,
  -0.18551318-0.06166904j, -0.18551318+0.06166904j],
 [ 0.37879116+0.j,          0.74196949+0.j,         0.74196949-0.j,
  -0.4083938 -0.01462004j, -0.4083938 +0.01462004j],
 [-0.17093882+0.j,         -0.18470612-0.48323373j, -0.18470612+0.48323373j,
  -0.04093935+0.01596304j, -0.04093935-0.01596304j],
 [ 0.00459468+0.j,          0.01156374+0.00194199j,  0.01156374-0.00194199j,
   0.88843853+0.j,          0.88843853-0.j,        ],
 [-0.00173699+0.j,         -0.00173499+0.00302986j, -0.00173499-0.00302986j,
   0.00259098+0.05934612j,  0.00259098-0.05934612j]])

# for misinformation bifurcation
evec_eq2_h2 = np.array([[-0.95254617 + 0.j, - 0.12970091 + 0.46407729j, - 0.12970091 - 0.46407729j,
          - 0.22641535 - 0.05594964j, - 0.22641535 + 0.05594964j],
         [0.28114392 + 0.j, 0.74782307 + 0.j, 0.74782307 - 0.j,
         - 0.40766072 - 0.01469347j, - 0.40766072 + 0.01469347j],
                    [-0.11660317 + 0.j, - 0.13727399 - 0.43548204j, - 0.13727399 + 0.43548204j,
                     - 0.03578366 + 0.01782442j, - 0.03578366 - 0.01782442j],
                    [0.00399095 + 0.j, 0.00855132 + 0.00223582j,
                     0.00855132 - 0.00223582j, 0.8810565 + 0.j, 0.8810565 - 0.j],
                    [-0.00128892 + 0.j, - 0.00111609 + 0.00157525j, - 0.00111609 - 0.00157525j,
                     0.00153302 + 0.0366262j,   0.00153302 - 0.0366262j]])






# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

ss_bp_r = {'x1': 0.00057159126, 'x2': 0.18949223,
        'x3': 0.19704689, 'x4': 0.60433083,
        'x5': 1}

par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

ss_hopf_r = {'x1': 0.107930, 'x2': 0.345919 ,
             'x3': 0.079105, 'x4': 0.393524,
             'x5': 0.001384}

eq1_lp1_ss = {'x1': 0.021799678852649853,
              'x2': 0.21186000608052885,
              'x3': 0.07439281159652406,
              'x4': 0.6784598802771113,
              'x5': 0.01111338310412665}

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

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



tend = 5_000 # in days end time for numerical simulation

def generate_ode(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = tend):
    """
    Function to generate an orinary differential equation for the system
    :param par_dict: dictionary for parameters
    :param ics_dict: dictionary for initial conditions
    :param tf: end time of simulation
    :return: generated ode for the system in PyDSTool
    """
    print('Generating ODE: executing ... generate_ode')

    # generate DSargs!
    DSargs = dst.args(name = 'infodemics_rev')

    # unpack parameters!!!!
    recovery = Par(par_dict['recovery'], 'recovery')
    belief = Par(par_dict['belief'], 'belief')
    risk = Par(par_dict['risk'], 'risk')
    protection = Par(par_dict['protection'], 'protection')
    education = Par(par_dict['education'], 'education')
    misinformation = Par(par_dict['misinformation'], 'misinformation')
    infection_good = Par(par_dict['infection_good'], 'infection_good')
    infection_bad = Par(par_dict['infection_bad'], 'infection_bad')

    DSargs.pars = [recovery, belief, risk, protection, education,
                   misinformation, infection_good, infection_bad]

    # generate state variables!
    x1 = Var('x1') # sg
    x2 = Var('x2') # sb
    x3 = Var('x3') # ib
    x4 = Var('x4') # v
    x5 = Var('x5') # phi

    # generate initial condiitons!
    DSargs.ics = ics_dict

    # generate time domain!
    DSargs.tdomain = [0, tf]

    # generate bounds on parameters!
    DSargs.pdomain = {'education': [0, 1], 'misinformation': [0, 1],
                      'infection_bad': [0, 1], 'infection_ bad': [0, 1],
                      'protection': [0, 1], 'risk': [0, 5],
                      'belief': [0, 1], 'recovery': [0, 1]}

    # generate bounds on state variables!
    DSargs.xdomain = {'x1': [0, 1], 'x2': [0, 1],
                      'x3': [0, 1], 'x4': [0, 1],
                      'x5': [0, 1]}

    # generate right hand side of the differential equations!
    x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (x5 + (infection_good + misinformation) * x3 + misinformation * x2)
    x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
    x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (1 - protection) * infection_good * x4)
    x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
    x5rhs = belief * x5 * (1 - x5) * (x3 + (1 - x1 - x2 - x3 - x4) - risk * x4)

    DSargs.varspecs = {'x1': x1rhs, 'x2': x2rhs,
                       'x3': x3rhs, 'x4': x4rhs,
                       'x5': x5rhs}

    DSargs.fnspecs = {'infected': (['t', 'sg', 'sb', 'v'],'1 - sg - sb - v'),
                       'bad': (['t', 'sb', 'ib'], 'sb + ib'),
                       'good': (['t', 'sb', 'ib', 'v'], '1 - sb - ib - v')}

    # change any default numerical parameters
    DSargs.algparams = {'max_pts': 10_000, 'stiff': True}

    # generate the ordinary differential equations!
    ode = dst.Vode_ODEsystem(DSargs)

    return ode

def generate_pointset(ODE = ode, save_bool = False, save_ss_data_name = 'infodemics_default_ss_data.txt', save_pts_data_name = 'infodemics_default_pts_data.txt'):
    """
    function to generate the pointset used for plotting temproal evolution and obtaining steady states!
    :param ode: instance of the generator from the PyDSTool in generate_ode argument!
    :param save_bool: boolean used to save the data
    :param save_pts_data_name: name for the saving of the data for the pointset
    :param save_ss_data_name: name for the saving of the data for the steady state
    :return: pointset generated from the ode
    """
    print('generating pointset!')
    # generate the pointset!
    pts = ode.compute('polarize').sample(dt = 5)
    sg_ss = pts['x1'][-1]
    sb_ss = pts['x2'][-1]
    ib_ss = pts['x3'][-1]
    v_ss = pts['x4'][-1]
    phi_ss = pts['x5'][-1]

    ss_dict = {'x1': sg_ss, 'x2': sb_ss,
               'x3': ib_ss, 'x4': v_ss, 'x5': phi_ss}

    # array's used for the saving of the data!
    ss_array = np.array([sg_ss, sb_ss, ib_ss, v_ss, phi_ss])
    pts_array = np.array([pts['x1'], pts['x2'], pts['x3'], pts['x4'], pts['x5'], pts['t']])

    print('Steady state values are: ')
    print(ss_dict)

    if save_bool == True:
        np.savetxt(save_ss_data_name, ss_array, delimiter = ',')
        np.savetxt(save_pts_data_name, pts_array, delimiter = ',')

    return pts, ss_dict


def plot_time_evolution(PTS = '', yvar = 'auxillary', load_bool = False):
    """
    function to generate a plotted figure!
    :param PTS: pointset from generate_pointset function
    :param yvar: y variable to be obtained in the plot: arguments are: auxillary, reff, state
    :return: plotted matplotlib figure!
    """
    if load_bool == False:
        pts = generate_pointset(ODE = ode)
        sg = pts['x1']
        sb = pts['x2']
        ib = pts['x3']
        v = pts['x4']
        phi = pts['x5']
        t = pts['t']
    else:
        pts_file_name = 'infodemics_default_pts_data.txt'
        pts = np.loadtxt(pts_file_name, delimiter = ',', dtype = float)
        sg = pts[:][0]
        sb = pts[:][1]
        ib = pts[:][2]
        v = pts[:][3]
        phi = pts[:][4]
        t = pts[:][5]


    infected = 1 - (sg + sb + v)
    good = 1 - (sb + ib + v)
    bad = ib + sb
    vaccinated = v

    if yvar == 'auxillary':
        plt.plot(t, infected, 'r', label = 'Infected')
        plt.plot(t, bad, 'k--', label = 'Bad')
        plt.plot(t, vaccinated, 'b:', label = 'Vaccinated')

    elif yvar == 'state':
        plt.plot(t, sb, 'r--', label = '$S_B$')
        plt.plot(t, ib, 'r', label = '$I_B$')
        plt.plot(t, v, 'b', label = '$V$')
        plt.plot(t, sg, 'b--', label = '$S_G$')
        plt.plot(t, phi, 'm:', label = '$\phi$')

    plt.legend()
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.show()



def generate_bifurcation(ODE = ode, PTS = '', SS_dict = '', par_dict = par_dict_def, par = 'risk', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, extend_par = 'protection', load_ss_bool = True, for_loop_bool = False, upper_branch_bool = False, max_n_points = 400):
    """
    function to generate a bifurcation with respect to the given 1 dimensional parameter
    :param ODE: system in generate_ode
    :param PTS: pointset in generate_pointset
    :param SS_dict: dictionary for steady state values used as initial condition
    :param par_dict: dictionary for parameters
    :param par: argument for the one-dimensional parameter bifurcation: use one of the parameters from DSargs.pars
    :param yvar: y-variable used for the y argument of the 1 dimensional bifurcation plot
    :param bif_type: bifurcation curve type: use EP-C, LP-C, etc. see PyCont documentation
    :param extend_bool: boolean to extend from a specific bifurcation on the previous plot to a co-dimension two bifurcation
    :param extend_par: parameter used for the co-dimension bifurcation provided the extend_bool is True!
    :param load_ss_bool: boolean to load the steady state data from the generate_pointset function
    :param for_loop_bool: boolean to use a for loop extension on the saddle node bifurcation -- use ext_par_bin in combination with par
    :param upper_branch_bool: boolean to extend along the upper branch
    :param max_n_points: maximum number of points!
    :return: matplotlib figure for generated bifurcation!
    """
    if par == 'risk':
        xlab = '$r$'
    elif par == 'education':
        xlab = r'$\tilde{\chi}_{gb}$'
    elif par == 'misinformation':
        xlab = r'$\tilde{\chi}_{bg}$'
    elif par == 'infection_good':
        xlab = r'$\chi_{gb}$'
    elif par == 'infection_bad':
        xlab = r'$\chi_{bb}$'
    elif par == 'protection':
        xlab = '$\delta$'
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

    if extend_par == 'risk':
        ylab_ext = '$r$'
    elif extend_par == 'education':
        ylab_ext = r'$\tilde{\chi}_{gb}$'
    elif extend_par == 'misinformation':
        ylab_ext = r'$\tilde{\chi}_{bg}$'
    elif extend_par == 'infection_good':
        ylab_ext = r'$\chi_{gb}$'
    elif extend_par == 'infection_bad':
        ylab_ext = r'$\chi_{bb}$'
    elif extend_par == 'protection':
        ylab_ext = '$\delta$'
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

    if yvar == 'x1':
        ylab = r'$S_G$'
    elif yvar == 'x2':
        yab = r'$S_B$'
    elif yvar == 'x3':
        ylab = r'$I_B$'
    elif yvar == 'x4':
        ylab = r'$Vaccinated$'
    elif yvar == 'x5':
        ylab = r'$\phi$'
    elif yvar == 'infected':
        ylab = '$Infected$'
    elif yvar == 'bad':
        ylab = '$Bad$'
    elif yvar ==  'good':
        ylab = '$Good$'
    else:
        ylab = ''
        print('choose yvar from one of the following:')
        print('\t x1')
        print('\t x2')
        print('\t x3')
        print('\t x4')
        print('\t x5')
        print('\t infected')
        print('\t bad')
        print('\t good')
        quit()


    # rename the function arguments!!!!!
    ode = ODE
    #ode = generate_ode(par_dict = par_dict_def, ics_dict = ics_dict_def)


    if load_ss_bool == True:
        ss_file_name = 'infodemics_default_ss_data.txt'
        ss = np.loadtxt(ss_file_name, delimiter=',', dtype=float)
        if SS_dict == '':
            ss_dict = {'x1': ss[0], 'x2': ss[1],
                       'x3': ss[2], 'x4': ss[3],
                       'x5': ss[4]}
        else:
            ss_dict = SS_dict

    else:
        pts, ss_dict = generate_pointset(ODE = ode)

    # use the initial conditions close to the steady state values
    ode.set(ics = ss_dict)
    ode.set(pars = par_dict)

    # generate a continuation from close to the steady state
    PC = ContClass(ode)

    if bif_type == 'EP-C':
        curve_name = 'EQ1'
    elif bif_type == 'LP-C':
        curve_name = 'HO1'
    else:
        curve_name = ''
        print('choose a type from one of the following:')
        print('\t EP-C')
        print('\t LP-C')
        quit()

    PCargs = dst.args(name = curve_name, type = bif_type)
    PCargs.freepars = [par] # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'
    # generate a numerical continuation
    PC.newCurve(PCargs)


    # continue backwards and forwards!
    PC[curve_name].forward()
    PC[curve_name].backward()

    # update the curve!
    # PCargs = dst.args(name='EQ2', type=bif_type)
    PCargs.freepars = [par]  # should be one of the parameters from DSargs.pars
    PCargs.initpoint = 'EQ1' + ':' + str('LP1')
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    PC.update(PCargs)
    PC['EQ1'].forward()
    PC['EQ1'].backward()

    # get the limit point and hopf bifurcation information


    lp1_dict = PC[curve_name].getSpecialPoint('LP1')
    h1_dict = PC[curve_name].getSpecialPoint('H1')
    b1_dict = PC[curve_name].getSpecialPoint('B1')


    yvar_array = ['x4', 'x3', 'x2']
    ylab_array = ['$V$', '$I_B$', '$S_B$']
    col_array = ['b', 'r', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display([par, yvar_array[z]], stability = True, color = col_array[z]) # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(xlab)
        plt.show()

    #PC.plot.toggleAll('off', bytype='P')

    # use the following to toggle any curve off!
    #PC.plot.toggleCurves('off', byname='HO1')

    #PCargs = dst.args(name='EQ2', type=bif_type)
    PCargs.freepars = [par]  # should be one of the parameters from DSargs.pars
    PCargs.initpoint = 'EQ1' + ':' + str('BP1')
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    PC.update(PCargs)
    PC['EQ1'].forward()
    PC['EQ1'].backward()

    # determine the following

    plt.title('') # no title
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.show()

    # generate an extended bifurcation for the 1D continuation

    # get the data along the continuation curve
    x1_sol = PC[curve_name].sol['x1']
    x2_sol = PC[curve_name].sol['x2']
    x3_sol = PC[curve_name].sol['x3']
    x4_sol = PC[curve_name].sol['x4']
    x5_sol = PC[curve_name].sol['x5']
    par_sol = PC[curve_name].sol[par]

    """# get the Hopf bifurcation on the lower branch to determine where the infection becomes stable and large!
    # use .getSpecialPoint() method
    special_point = 'H1'
    eq1_hopf1 = PC[curve_name].getSpecialPoint(special_point) # remark: this is a dictionary!
    par_hopf = eq1_hopf1[par]"""

    """special_point = 'B1'
    eq1_b1 = PC[curve_name].getSpecialPoint(special_point)  # remark: this is a dictionary!
    par_b1 = eq1_b1[par]"""

    #print(f"{special_point} Information about parameter: ", eq1_hopf1)

    # determine the smallest index closest to the hopf bifurcation
    #idx_unstable = np.argmin(np.abs(par_sol - np.ones(len(par_sol)) * par_hopf))


    if extend_bool:

        # ask user for special point

        # initialize the parameters at the initial limit point!
        eq1_lp1_dict = PC[curve_name].getSpecialPoint('LP1')
        eq1_lp1_ss_dict = {'x1': eq1_lp1_dict['x1'], 'x2': eq1_lp1_dict['x2'],
                           'x3': eq1_lp1_dict['x3'], 'x4': eq1_lp1_dict['x4'],
                           'x5': eq1_lp1_dict['x5']}















def generate_nullclines(ODE = ode, ss_dict = '', pts = '', xvar = 'x1', yvar = 'x3', load_data_bool = False):
    """
    function to generate nullclines in the phase space
    :param ODE: ode from gen_ode
    :param ss_dict: dictionary for steady state values (must be a fixed point)
    :param pts: pointset from generate_pointset
    :param xvar: variable along the x-axis
    :param yvar: variable along the y-axis
    :param load_data_bool: boolean value to load the data for the pts and ss_dict
    :return: matplotlib figures for the nullclines with trajectory
    """
    ode = ODE
    if load_data_bool == True:
        ss_file_name = 'infodemics_default_ss_data.txt'
        pts_file_name = 'infodemics_default_pts_data.txt'

        ss_data = np.loadtxt(ss_file_name, delimiter = ',', dtype = float)
        pts = np.loadtxt(pts_file_name, delimiter = ',', dtype = float)

        # convert steady state array to steady state dictionary
        ss_dict = {'x1': ss_data[0], 'x2': ss_data[1],
                   'x3': ss_data[2], 'x4': ss_data[3],
                   'x5': ss_data[4]}

        # obtain the trajectory of the variables
        x1 = pts[:][0]
        x2 = pts[:][1]
        x3 = pts[:][2]
        x4 = pts[:][3]
        x5 = pts[:][4]
        t = pts[:][5]

    else:
        # generate pointset and steady state dictionary
        pts, ss_dict = generate_pointset(ODE = ode)

        x1 = pts['x1']
        x2 = pts['x2']
        x3 = pts['x3']
        x4 = pts['x4']
        x5 = pts['x5']
        t = pts['t']

    fp_coord = ss_dict # used for fixed point about the nullclines

def sys_dx(X, t = 0, par_dict = par_bp_r, ss_dict = ss_bp_r, xvar = 'x1', yvar  = 'x3'):
    """
    function to generate the phase field of the state varialbes and parameters
    :param X: 5 dimensional array
    :param t: time
    :param par_dict: dicitonary of parameters
    :param ss_dict: dictioanry of steady state variables
    :param xvar: x variable for the phase space
    :param yvar: y variable for the phase space
    :return: Z, the array of the data
    """
    x1_ss = ss_dict['x1']
    x2_ss = ss_dict['x2']
    x3_ss = ss_dict['x3']
    x4_ss = ss_dict['x4']
    x5_ss = ss_dict['x5']

    # unpack parameters
    risk = par_dict['risk']
    protection = par_dict['protection']
    belief = par_dict['belief']
    infection_bad = par_dict['infection_bad']
    infection_good = par_dict['infection_good']
    misinformation = par_dict['misinformation']
    education = par_dict['education']
    recovery = par_dict['recovery']

    if xvar == 'x1':
        x1 = X[0]
    else:
        x1 = x1_ss
    if xvar == 'x2':
        x2 = X[0]
    else:
        x2 = x2_ss
    if xvar == 'x3':
        x3 = x3_ss
    else:
        x3 = X[0]
    if xvar == 'x4':
        x4 = X[0]
    else:
        x4 = x4_ss
    if xvar == 'x5':
        x5 = X[0]
    else:
        x5 = x5_ss

    # -- y variable
    if yvar == 'x1':
        x1 = X[1]
    else:
        x1 = x1_ss
    if yvar == 'x2':
        x2 = X[1]
    else:
        x2 = x2_ss
    if yvar == 'x3':
        x3 = X[1]
    else:
        x3 = x3_ss
    if yvar == 'x4':
        x4 = X[1]
    else:
        x4 = x4_ss
    if yvar == 'x5':
        x5 = X[1]
    else:
        x5 = x5_ss


    # generate right hand side of the differential equations!
    x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (
                x5 + (infection_good + misinformation) * x3 + misinformation * x2)
    x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
    x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (
                1 - protection) * infection_good * x4)
    x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
    x5rhs = belief * x5 * (1 - x5) * (x3 + (1 - x1 - x2 - x3 - x4) - risk * x4)

    if xvar == 'x1':
        v1 = x1rhs
    elif xvar == 'x2':
        v1 = x2rhs
    elif xvar == 'x3':
        v1 = x3rhs
    elif xvar == 'x4':
        v1 = x4rhs
    elif xvar == 'x5':
        v1 = x5rhs
    else:
        v1 = 0

    if yvar == 'x1':
        v2 = x1rhs
    elif yvar == 'x2':
        v2 = x2rhs
    elif yvar == 'x3':
        v2 = x3rhs
    elif yvar == 'x4':
        v2 = x4rhs
    elif yvar == 'x5':
        v2 = x5rhs
    else:
        v2 = 0

    Z = np.array([v1, v2])
    return Z

def plot_nullclines(option = 'A', par_dict = eq1_h1_par_dict, ss_dict = eq1_h1_ss, n_bin = 100, xlow = 0, xhigh = 1.0, ylow = 0, yhigh = 1.0, quiv_bool = True, ext_bool = False, ss_dict_2 = eq1_lp1_ss, par_dict_2 = eq1_lp1_par_dict, evecs_bool = False, evecs = None):
    """
    function to generate nullclines with a quiver plot field determined using sys_dx function
    :param option: 'A' for SG vs IB, 'B' for SB vs IB, 'C' for IB vs V
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

    x1_0 = 0.20
    x2_0 = 0.50
    x3_0 = 0.001
    x4_0 = 0.14
    x5_0 = 0.0003

    ics = {'x1': x1_0, 'x2': x2_0,
           'x3': x3_0, 'x4': x4_0, 'x5': x5_0}

    ode = generate_ode(par_dict=par_dict, ics_dict=ics, tf=10_000)
    pts = ode.compute('nulls_traj').sample(dt = 5)
    #pts, ss_dict = generate_pointset(ODE = ode, save_bool=False)
    x1_traj = pts['x1']
    x2_traj = pts['x2']
    x3_traj = pts['x3']
    x4_traj = pts['x4']
    x5_traj = pts['x5']
    t = pts['t']

    # determine the functions of nullclines
    if option == 'A':
        # sg vs sb
        a = 0 # eigenvector component in x
        b = 1 # eigenvector component in y

        xlab = r'$S_G$' # label for x axis
        ylab = r'$S_B$' # label for y axis

        xnull_lab = r'$N_{S_G}$' # label for legend
        ynull_lab = r'$N_{S_B}$'

        x_traj = x1_traj
        y_traj = x2_traj

        dx_x = 'x1' # used for quiver
        dx_y = 'x2'

        x_ss = x1_ss # steady state value on x
        y_ss = x2_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x2
        x_null = (recovery * ig_ss - x_array * (x5_ss + (infection_good + misinformation) * x3_ss)) / (misinformation * x_array)
        # y null --> solve for x = x1
        y_null = (x3_ss * (infection_bad * y_array - recovery)) / (misinformation * (y_array + x3_ss))

    elif option == 'B':
        # sb vs ib
        a = 1 # eigenvector component in x
        b = 2 # eigenvector component in y

        xlab = r'$S_B$' # label for x axis
        ylab = r'$I_B$' # label for y axis

        xnull_lab = r'$N_{S_B}$' # label for legend
        ynull_lab = r'$N_{I_B}$'

        x_traj = x2_traj
        y_traj = x3_traj

        dx_x = 'x2' # used for quiver
        dx_y = 'x3'

        x_ss = x2_ss # steady state value on x
        y_ss = x3_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x3
        x_null = (misinformation * x1_ss * x_array) / (infection_bad * x_array - recovery - misinformation * x1_ss)
        # y null --> solve for x = x2
        y_null = (recovery + education * (x1_ss + ig_ss) - x5_ss * x1_ss / y_array) / infection_bad

    elif option == 'C':
        # ib vs v
        a = 2 # eigenvector component in x
        b = 3 # eigenvector component in y

        xlab = r'$I_B$' # label for x axis
        ylab = r'$V$' # label for y axis

        xnull_lab = r'$N_{I_B}$' # label for legend
        ynull_lab = r'$N_{V}$'

        x_traj = x3_traj
        y_traj = x4_traj

        dx_x = 'x3' # used for quiver
        dx_y = 'x4'

        x_ss = x3_ss # steady state value on x
        y_ss = x4_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        #x_null = (misinformation * x2_ss - recovery + education * (x2_ss + x_array - 1)) / ((protection - 1) * infection_good)
        x_null = np.ones(n_bin) * (recovery + education * (x1_ss + ig_ss) - infection_bad * x2_ss) / ((1  - protection) * infection_good)
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
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict = par_dict, ss_dict = ss_dict, xvar = dx_x, yvar = dx_y)
        # normalize growth rate!
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1 # avoid division of zero
        dx1 /= M
        dy1 /= M # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot = 'mid')

    z = int(len(x1_traj) / 4)
    plt.plot(x_array, x_null, 'b', label = xnull_lab)
    plt.plot(y_null, y_array, 'r', label = ynull_lab)

    if dx_x == 'x3':
        plt.vlines(x = 0, ymin = ylow, ymax = yhigh, color = 'b')
    if dx_y == 'x3':
        plt.hlines(y = 0, xmin = xlow, xmax = xhigh, color = 'r')

    # plot steady state values!
    plt.plot(x_ss, y_ss, 'ko', fillstyle = 'left', ms = 10)

    # plot eigenvectors, if boolean is true
    if evecs_bool == True:
        evec_real = evecs.real # real part of the eigenvalue
        # determine the components to use of the eigenvalue depending on the x and y values used for the nullclines
        v1 = np.array([evec_real[a][a], evec_real[b][a]])
        v2 = np.array([evec_real[b][a], evec_real[b][b]])

        # determine the angle between the eigenvectors
        evecs_angle = np.arccos(np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

        # plot the eigenvector's starting from the fied point
        plt.arrow(x_ss, y_ss, v1[0], v1[1], color = 'b', ls = '--')
        plt.arrow(x_ss, y_ss, v2[0], v2[1], color = 'r', ls = '--')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.plot(x_traj[z:], y_traj[z:], 'k')
    plt.legend()
    plt.show()


    # time plot
    plt.plot(t, x1_traj, 'b-', label = '$S_G$')
    plt.plot(t, x2_traj, 'r--', label = '$S_B$')
    plt.plot(t, x3_traj, 'r', label='$I_B$')
    plt.plot(t, x4_traj, 'b:', label = '$V$')
    plt.plot(t, x5_traj, 'm:', label='$\phi$')
    plt.legend()
    plt.xlabel('t (Days)')
    plt.ylabel('Fraction of the Population')
    plt.show()


def generate_bifurcation_2d(PC, curve='EQ1', special_point='H1', xpar='misinformation', ypar='education',
                            max_n_points=80, curve_type='H-C1', par_dict=eq2_h2_par):
    """
    funciton to generate a co-dimension two bifurcation given an initial starting point on the curve
    :param PC: PyCont generated from generate_bifurcation
    :param curve: curve created from the generate_bifurcation plot
    :param special_point: LP, HO, BP, B, etc. special points determined from the user input for the previous curve
    :param xpar: parameter for the x value (original parameter used for the codimension 1 curve bifurcation)
    :param ypar: parameter for the y value (extension from the codimension 1 bifurcation previously)
    :param max_n_points: maximum number of points
    :param curve_type: Use: H-C1, H-C2, LP-C
    :param par_dict: dictionary of parameters used for the initialization of continuation
    :return: plotted bifurcation via matplotlib
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
        xlab = r'$\chi_{bb}$'
    elif xpar == 'protection':
        xlab = '$\delta$'
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
        ylab = r'$\chi_{bb}$'
    elif ypar == 'protection':
        ylab = '$\delta$'
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

    # grab the data from the user supplied special point
    point_special = PC[curve].getSpecialPoint(special_point)

    # get the steady state values associated to the special point!
    ss_dict = {'x1': point_special['x1'],
               'x2': point_special['x2'],
               'x3': point_special['x3'],
               'x4': point_special['x4'],
               'x5': point_special['x5']}

    par_dict[xpar] = point_special[xpar]  # assign the parameter value at the dictionary value of the special point

    # now we generate a continuation!
    PCargs = dst.args(name='HO1', type=curve_type)
    PCargs.freepars = [xpar, ypar]  # should be one of the parameters from DSargs.pars --
    PCargs.initpoint = ss_dict # dictionary for steady state values!
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations / hopf bifurcations!
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'  # boundary for the continuation
    # generate a numerical continuation
    PC.newCurve(PCargs)

    # continue backwards and forwards!
    PC['HO1'].forward()
    PC['HO1'].backward()

    PC['HO1'].forward()
    PC['HO1'].backward()

    # now we display the continuation curve with respect to the variables!
    PC.display([xpar, ypar])

    # display the bifurcation!
    PC.display([xpar, ypar], stability=True)  # x variable vs y variable!
    # disable the boundary
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
    plt.title('')  # no title
    plt.ylabel(ylab_ext)
    plt.xlabel(xlab)
    plt.show()




# Arguments for the bifurcation system!

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

ss_bp_r = {'x1': 0.00057159126, 'x2': 0.18949223,
        'x3': 0.19704689, 'x4': 0.60433083,
        'x5': 1}

par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

ss_hopf_r = {'x1': 0.107930, 'x2': 0.345919 ,
             'x3': 0.079105, 'x4': 0.393524,
             'x5': 0.001384}

eq1_lp1_ss = {'x1': 0.021799678852649853,
              'x2': 0.21186000608052885,
              'x3': 0.07439281159652406,
              'x4': 0.6784598802771113,
              'x5': 0.01111338310412665}

# risk bifurcation hopf bifurcaiton on the lower branch
eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

# this one phi is out of bounds!
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


# this is the steady state value for misinformation starting from the hopf on the risk bifurcation
eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927
tend = 5_000 # in days end time for numerical simulation



eq1_h1_ss_new = {'x1': 0.20993799300477495,
                   'x2': 0.48221092545757344,
                   'x3': 0.07073195797121161,
                   'x4': 0.1168184171123022,
                   'x5': 0.00018891967673741587}

eq1_h1_par_dict_new = par_bp_r
eq1_h1_par_dict_new['risk'] = 1.635295791362042


# PLOT specific nullclines!
def plot_sg_sb_nullcline(par_dict = eq1_h1_par_dict, ss_dict = eq1_h1_ss_new, xhigh = 0.65, yhigh = 0.65):
    """
    function to generate the sg vs sb nullcline
    :param par_dict: dictionary for parameters used to obtain the steady states
    :param ss_dict: dictionary for steady states
    :param xhigh: upper limit on the x variable
    :param yhigh: upper limit on the y variable
    :return: the plotted bifurcaiton
    """
    # generate_bifurcation(ODE = ode, PTS = '', SS_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, par = 'risk', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, extend_par = 'protection', load_ss_bool = True, for_loop_bool = False, max_n_points = 100)
    # gives an output of:
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

    plot_nullclines(ss_dict=ss_dict, par_dict=par_dict, option = 'A', xhigh = xhigh, yhigh = yhigh)

def plot_sb_ib_nullcline(par_dict = eq1_h1_par_dict, ss_dict = eq1_h1_ss_new, xhigh = 0.65, yhigh = 0.65):
    """
    function to generate the sb vs ib nullcline
    :param par_dict: dictionary for parameters used to obtain the steady states
    :param ss_dict: dictionary for steady states
    :param xhigh: upper limit on the x variable
    :param yhigh: upper limit on the y variable
    :return: the plotted bifurcaiton
    """
    plot_nullclines(ss_dict=ss_dict, par_dict=par_dict, option = 'B', xhigh = xhigh, yhigh = yhigh)

def plot_ib_v_nullcline(par_dict = eq1_h1_par_dict, ss_dict = eq1_h1_ss_new, xhigh = 0.65, yhigh = 0.65):
    """
    function to generate the ib vs v nullcline
    :param par_dict: dictionary for parameters used to obtain the steady states
    :param ss_dict: dictionary for steady states
    :param xhigh: upper limit on the x variable
    :param yhigh: upper limit on the y variable
    :return: the plotted bifurcaiton
    """
    plot_nullclines(ss_dict=ss_dict, par_dict=par_dict, option = 'C', xhigh = xhigh, yhigh = yhigh)



# generate the ode
ode = generate_ode(eq1_h1_par_dict, ics_dict = eq1_h1_ss_new)

# generate a pointset!
pts, ss_dict = generate_pointset(save_bool = True)

# plot time!
plot_time_evolution(yvar = 'state', load_bool=True)

# plot bifurcation!
PC = generate_bifurcation(ODE = ode, PTS = '', SS_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, par = 'risk', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, extend_par = 'protection', load_ss_bool = True, for_loop_bool = False, max_n_points = 200)

# Plotting sg vs sb nullcline!
plot_sg_sb_nullcline()
plot_sb_ib_nullcline()
plot_ib_v_nullcline()


# BRANCHING POINT!
# risk bifurcation on EQ!
"""BP Point found 
========================== 
0 : 
x1  =  0.0005714445346933421
x2  =  0.18938936824988012
x3  =  0.1968691906600376
x4  =  0.6047210552785887
x5  =  1.0
risk  =  0.33952535659974636
"""
eq_risk_bp_ss = {'x1': 0.0005714445346933421, 
                 'x2': 0.1893893682498801, 
                 'x3': 0.1968691906600376,
                 'x4': 0.6047210552785887,
                 'x5': 1.0}

eq_risk_bp_par = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# generate a bifurcation and extend along the branching point



#plot_nullclines(ss_dict=eq2_h2_ss, par_dict=eq2_h2_par, evecs = evec_eq2_h2, evecs_bool = True, option = 'A', xhigh = 0.65)
#plot_nullclines(ss_dict=eq2_h2_ss, par_dict=eq2_h2_par, evecs = evec_eq2_h2, evecs_bool = True, option = 'B', xhigh = 0.65)
#plot_nullclines(ss_dict=eq2_h2_ss, par_dict=eq2_h2_par, evecs = evec_eq2_h2, evecs_bool = True, option = 'C', xhigh = 0.65)


#plot_nullclines(ss_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, evecs = evec_eq1_h1, evecs_bool = True, option = 'A', xhigh = 0.65)
#plot_nullclines(ss_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, evecs = evec_eq1_h1, evecs_bool = True, option = 'B', xhigh = 0.65)
#plot_nullclines(ss_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, evecs = evec_eq1_h1, evecs_bool = True, option = 'C', xhigh = 0.65)






#PC = generate_bifurcation(ODE = ode, PTS = '', SS_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, par = 'misinformation', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, extend_par = 'education', load_ss_bool = True, for_loop_bool = False, max_n_points = 55)

#generate_bifurcation_2d(PC)


# In[ ]:


# revise the bifurcation generation along the plot

def generate_bifurcation(ODE = ode, PTS = '', SS_dict = '', par_dict = par_dict_def, par = 'risk', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, load_ss_bool = True, max_n_points = 400):
    """
    function to generate a bifurcation with respect to the given 1 dimensional parameter
    :param ODE: system in generate_ode
    :param PTS: pointset in generate_pointset
    :param SS_dict: dictionary for steady state values used as initial condition
    :param par_dict: dictionary for parameters
    :param par: argument for the one-dimensional parameter bifurcation: use one of the parameters from DSargs.pars
    :param yvar: y-variable used for the y argument of the 1 dimensional bifurcation plot
    :param bif_type: bifurcation curve type: use EP-C, LP-C, etc. see PyCont documentation
    :param extend_bool: boolean to extend from a specific bifurcation on the previous plot to a co-dimension two bifurcation
    :param extend_par: parameter used for the co-dimension bifurcation provided the extend_bool is True!
    :param load_ss_bool: boolean to load the steady state data from the generate_pointset function
    :param max_n_points: maximum number of points!
    :return: matplotlib figure for generated bifurcation!
    """
    if par == 'risk':
        xlab = '$r$'
    elif par == 'education':
        xlab = r'$\tilde{\chi}_{gb}$'
    elif par == 'misinformation':
        xlab = r'$\tilde{\chi}_{bg}$'
    elif par == 'infection_good':
        xlab = r'$\chi_{gb}$'
    elif par == 'infection_bad':
        xlab = r'$\chi_{bb}$'
    elif par == 'protection':
        xlab = '$\delta$'
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

    if extend_par == 'risk':
        ylab_ext = '$r$'
    elif extend_par == 'education':
        ylab_ext = r'$\tilde{\chi}_{gb}$'
    elif extend_par == 'misinformation':
        ylab_ext = r'$\tilde{\chi}_{bg}$'
    elif extend_par == 'infection_good':
        ylab_ext = r'$\chi_{gb}$'
    elif extend_par == 'infection_bad':
        ylab_ext = r'$\chi_{bb}$'
    elif extend_par == 'protection':
        ylab_ext = '$\delta$'
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

    if yvar == 'x1':
        ylab = r'$S_G$'
    elif yvar == 'x2':
        yab = r'$S_B$'
    elif yvar == 'x3':
        ylab = r'$I_B$'
    elif yvar == 'x4':
        ylab = r'$Vaccinated$'
    elif yvar == 'x5':
        ylab = r'$\phi$'
    elif yvar == 'infected':
        ylab = '$Infected$'
    elif yvar == 'bad':
        ylab = '$Bad$'
    elif yvar ==  'good':
        ylab = '$Good$'
    else:
        ylab = ''
        print('choose yvar from one of the following:')
        print('\t x1')
        print('\t x2')
        print('\t x3')
        print('\t x4')
        print('\t x5')
        print('\t infected')
        print('\t bad')
        print('\t good')
        quit()


    # rename the function arguments!!!!!
    ode = ODE
    #ode = generate_ode(par_dict = par_dict_def, ics_dict = ics_dict_def)


    if load_ss_bool == True:
        ss_file_name = 'infodemics_default_ss_data.txt'
        ss = np.loadtxt(ss_file_name, delimiter=',', dtype=float)
        if SS_dict == '':
            ss_dict = {'x1': ss[0], 'x2': ss[1],
                       'x3': ss[2], 'x4': ss[3],
                       'x5': ss[4]}
        else:
            ss_dict = SS_dict

    else:
        pts, ss_dict = generate_pointset(ODE = ode)

    # use the initial conditions close to the steady state values
    ode.set(ics = ss_dict)
    ode.set(pars = par_dict)

    # generate a continuation from close to the steady state
    PC = ContClass(ode)

    if bif_type == 'EP-C':
        curve_name = 'EQ1'
    elif bif_type == 'LP-C':
        curve_name = 'HO1'
    else:
        curve_name = ''
        print('choose a type from one of the following:')
        print('\t EP-C')
        print('\t LP-C')
        quit()

    PCargs = dst.args(name = curve_name, type = bif_type)
    PCargs.freepars = [par] # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'
    # generate a numerical continuation
    PC.newCurve(PCargs)


    # continue backwards and forwards!
    PC[curve_name].forward()
    PC[curve_name].backward()

    # update the curve!
    # PCargs = dst.args(name='EQ2', type=bif_type)
    PCargs.freepars = [par]  # should be one of the parameters from DSargs.pars
    PCargs.initpoint = 'EQ1' + ':' + str('LP1')
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    PC.update(PCargs)
    PC['EQ1'].forward()
    PC['EQ1'].backward()
    
    
    # provided the extend_bool = True, then we want to choose one of the bifurcation points and update it

    # get the limit point and hopf bifurcation information


    lp1_dict = PC[curve_name].getSpecialPoint('LP1')
    h1_dict = PC[curve_name].getSpecialPoint('H1')
    b1_dict = PC[curve_name].getSpecialPoint('B1')


    yvar_array = ['x4', 'x3', 'x2']
    ylab_array = ['$V$', '$I_B$', '$S_B$']
    col_array = ['b', 'r', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display([par, yvar_array[z]], stability = True, color = col_array[z]) # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(xlab)
        plt.show()

    #PC.plot.toggleAll('off', bytype='P')

    # use the following to toggle any curve off!
    #PC.plot.toggleCurves('off', byname='HO1')

    #PCargs = dst.args(name='EQ2', type=bif_type)
    PCargs.freepars = [par]  # should be one of the parameters from DSargs.pars
    PCargs.initpoint = 'EQ1' + ':' + str('BP1')
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    PC.update(PCargs)
    PC['EQ1'].forward()
    PC['EQ1'].backward()

    # determine the following

    plt.title('') # no title
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.show()

    # generate an extended bifurcation for the 1D continuation

    # get the data along the continuation curve
    x1_sol = PC[curve_name].sol['x1']
    x2_sol = PC[curve_name].sol['x2']
    x3_sol = PC[curve_name].sol['x3']
    x4_sol = PC[curve_name].sol['x4']
    x5_sol = PC[curve_name].sol['x5']
    par_sol = PC[curve_name].sol[par]

    """# get the Hopf bifurcation on the lower branch to determine where the infection becomes stable and large!
    # use .getSpecialPoint() method
    special_point = 'H1'
    eq1_hopf1 = PC[curve_name].getSpecialPoint(special_point) # remark: this is a dictionary!
    par_hopf = eq1_hopf1[par]"""

    """special_point = 'B1'
    eq1_b1 = PC[curve_name].getSpecialPoint(special_point)  # remark: this is a dictionary!
    par_b1 = eq1_b1[par]"""

    #print(f"{special_point} Information about parameter: ", eq1_hopf1)

    # determine the smallest index closest to the hopf bifurcation
    #idx_unstable = np.argmin(np.abs(par_sol - np.ones(len(par_sol)) * par_hopf))


    if extend_bool:

        # ask user for special point

        # initialize the parameters at the initial limit point!
        eq1_lp1_dict = PC[curve_name].getSpecialPoint('LP1')
        eq1_lp1_ss_dict = {'x1': eq1_lp1_dict['x1'], 'x2': eq1_lp1_dict['x2'],
                           'x3': eq1_lp1_dict['x3'], 'x4': eq1_lp1_dict['x4'],
                           'x5': eq1_lp1_dict['x5']}



eq_risk_bp_ss = {'x1': 0.0005714445346933421, 
                 'x2': 0.1893893682498801, 
                 'x3': 0.1968691906600376,
                 'x4': 0.6047210552785887,
                 'x5': 1.0}

eq_risk_bp_par = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

PC = generate_bifurcation(ODE = ode, PTS = '', SS_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, par = 'risk', yvar = 'x4', bif_type = 'EP-C', extend_bool = False, load_ss_bool = True, max_n_points = 200)


# In[ ]:


def generate_risk_bifurcation(ODE = ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 250, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    ode = generate_ode(par_dict, ics_dict, tf = tend)    # generate a pointset
    #pts, ss_dict = generate_pointset(ode)
    
    # use the initial conditions at the steady state
    #ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQrisk', type = 'EP-C')
    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 1e-4
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'
    # generate a numerical continuation
    PC.newCurve(PCargs)

    # continue backwards and forwards!
    PC['EQrisk'].forward()
    PC['EQrisk'].backward()

    # update the curve along the branching point!
    """eq_risk_bp_ss = {'x1': 0.0005714445346933421, 
                 'x2': 0.1893893682498801, 
                 'x3': 0.1968691906600376,
                 'x4': 0.6047210552785887,
                 'x5': 1.0}

    eq_risk_bp_par = {'recovery': 0.07, 'belief': 1.0,
                'risk': 0.34021985, 'protection': 0.90,
                'education': 0.33, 'misinformation': 0.10,
                'infection_good': 0.048, 'infection_bad': 0.37}
    ode.set(ics = eq_risk_bp_ss,
           pars = eq_risk_bp_par)"""
    
    """PCargs.initpoint = {'x1': 0.0005714445346933421, 
                 'x2': 0.1893893682498801, 
                 'x3': 0.1968691906600376,
                 'x4': 0.6047210552785887,
                 'x5': 1.0, 
                 'risk': 0.0}"""
    
    PCargs.name = 'EQrisk2'
    PCargs.type = 'EP-C'
    PCargs.initpoint = {'x1': 0.0005714445346933421, 
                 'x2': 0.1893893682498801, 
                 'x3': 0.1968691906600376,
                 'x4': 0.6047210552785887,
                 'x5': 1.0, 
                 'risk': 0.0}
    
    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.0*max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    #PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQrisk2'].forward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$']
    col_array = ['b', 'r', 'k', 'orange']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['risk', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$r$')
        plt.show()
        
    return PC


# In[272]:


eq1_h1_par_dict
eq1_h1_par_dict['misinformation'] = 0.10 
eq1_h1_par_dict
eq1_h1_ss


# In[273]:


#PC2 = generate_risk_bifurcation(ics_dict = ss_bp_r, par_dict = par_bp_r)
PC = generate_risk_bifurcation(ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, tend = 300)


# In[274]:


def plot_nullclines_new(option = 'A', PTS = '', par_dict = eq1_h1_par_dict, ss_dict = eq1_h1_ss, n_bin = 100, xlow = 0, xhigh = 1.0, ylow = 0, yhigh = 1.0, quiv_bool = True, ext_bool = False, ss_dict_2 = eq1_lp1_ss, par_dict_2 = eq1_lp1_par_dict, evecs_bool = False, evecs = None, ics_dict = {}):
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
        pts = ode.compute('nulls_traj').sample(dt = 5)
    else:
        pts = PTS
    #pts, ss_dict = generate_pointset(ODE = ode, save_bool=False)
    x1_traj = pts['x1']
    x2_traj = pts['x2']
    x3_traj = pts['x3']
    x4_traj = pts['x4']
    x5_traj = pts['x5']
    t = pts['t']

    # determine the functions of nullclines
    if option == 'A' or option == '':
        # sg vs sb
        a = 0 # eigenvector component in x
        b = 1 # eigenvector component in y

        xlab = r'$S_G$' # label for x axis
        ylab = r'$S_B$' # label for y axis

        xnull_lab = r'$N_{S_G}$' # label for legend
        ynull_lab = r'$N_{S_B}$'

        x_traj = x1_traj
        y_traj = x2_traj

        dx_x = 'x1' # used for quiver
        dx_y = 'x2'

        x_ss = x1_ss # steady state value on x
        y_ss = x2_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x2
        x_null = (recovery * ig_ss - x_array * (x5_ss + (infection_good + misinformation) * x3_ss)) / (misinformation * x_array)
        # y null --> solve for x = x1
        y_null = (x3_ss * (infection_bad * y_array - recovery)) / (misinformation * (y_array + x3_ss))

    elif option == 'B':
        # sb vs ib
        a = 1 # eigenvector component in x
        b = 2 # eigenvector component in y

        xlab = r'$S_B$' # label for x axis
        ylab = r'$I_B$' # label for y axis

        xnull_lab = r'$N_{S_B}$' # label for legend
        ynull_lab = r'$N_{I_B}$'

        x_traj = x2_traj
        y_traj = x3_traj

        dx_x = 'x2' # used for quiver
        dx_y = 'x3'

        x_ss = x2_ss # steady state value on x
        y_ss = x3_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x3
        x_null = (misinformation * x1_ss * x_array) / (infection_bad * x_array - recovery - misinformation * x1_ss)
        # y null --> solve for x = x2
        y_null = (recovery + education * (x1_ss + ig_ss) - x5_ss * x1_ss / y_array) / infection_bad

    elif option == 'C':
        # ib vs v
        a = 2 # eigenvector component in x
        b = 3 # eigenvector component in y

        xlab = r'$I_B$' # label for x axis
        ylab = r'$V$' # label for y axis

        xnull_lab = r'$N_{I_B}$' # label for legend
        ynull_lab = r'$N_{V}$'

        x_traj = x3_traj
        y_traj = x4_traj

        dx_x = 'x3' # used for quiver
        dx_y = 'x4'

        x_ss = x3_ss # steady state value on x
        y_ss = x4_ss # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        #x_null = (misinformation * x2_ss - recovery + education * (x2_ss + x_array - 1)) / ((protection - 1) * infection_good)
        x_null = np.ones(n_bin) * (recovery + education * (x1_ss + ig_ss) - infection_bad * x2_ss) / ((1  - protection) * infection_good)
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
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict = par_dict, ss_dict = ss_dict, xvar = dx_x, yvar = dx_y)
        # normalize growth rate!
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1 # avoid division of zero
        dx1 /= M
        dy1 /= M # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot = 'mid')

    z = int(len(x1_traj) / 4)
    plt.plot(x_array, x_null, 'b', label = xnull_lab)
    plt.plot(y_null, y_array, 'r', label = ynull_lab)

    if dx_x == 'x3':
        plt.vlines(x = 0, ymin = ylow, ymax = yhigh, color = 'b')
    if dx_y == 'x3':
        plt.hlines(y = 0, xmin = xlow, xmax = xhigh, color = 'r')

    # plot steady state values!
    plt.plot(x_ss, y_ss, 'ko', fillstyle = 'left', ms = 10)

    # plot eigenvectors, if boolean is true
    if evecs_bool == True:
        evec_real = evecs.real # real part of the eigenvalue
        # determine the components to use of the eigenvalue depending on the x and y values used for the nullclines
        v1 = np.array([evec_real[a][a], evec_real[b][a]])
        v2 = np.array([evec_real[b][a], evec_real[b][b]])

        # determine the angle between the eigenvectors
        evecs_angle = np.arccos(np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

        # plot the eigenvector's starting from the fied point
        plt.arrow(x_ss, y_ss, v1[0], v1[1], color = 'b', ls = '--')
        plt.arrow(x_ss, y_ss, v2[0], v2[1], color = 'r', ls = '--')

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.plot(x_traj[0:], y_traj[0:], 'k')
    plt.legend()
    plt.show()


    # time plot
    plt.plot(t, x1_traj, 'b-', label = '$S_G$')
    plt.plot(t, x2_traj, 'r--', label = '$S_B$')
    plt.plot(t, x3_traj, 'r', label='$I_B$')
    plt.plot(t, x4_traj, 'b:', label = '$V$')
    plt.plot(t, x5_traj, 'm:', label='$\phi$')
    plt.legend()
    plt.xlabel('t (Days)')
    plt.ylabel('Fraction of the Population')
    plt.show()


# In[275]:


PC['EQrisk'].info()


# In[276]:


PC['EQrisk'].getSpecialPoint('H1')


# In[277]:


PC['EQrisk'].getSpecialPoint('LP1')


# In[278]:


eqrisk_bp1 = PC['EQrisk2'].getSpecialPoint('BP1')


# In[242]:


#eqrisk_h1 = PC['EQrisk'].getSpecialPoint('H1')
#eqrisk_h1
eqrisk_bp1 = PC['EQrisk2'].getSpecialPoint('BP1')


def get_data(PC, curve = 'EQrisk', special_point = 'H1', par = 'risk', par_ext = '', par_dict = eq1_h1_par_dict):
    """
    function to get_data from the bifurcation plot!
    :param curve: PC[curve]
    :param special_point: PC[curve.getSpecialPoint(special_point)
    :param par_ext: if '' do not use, otherwise there will be an additional parameter
    :param par_dict, dictionary for parameters used in the bifurcation as the baseline
    :return: par_dict, ss_dict <--> dictionary of parameters and steady state values
    """
    data = PC[curve].getSpecialPoint(special_point)
    par_dict[par] = data[par]
    
    if par_ext != '':
        par_dict[par_ext] = data[par_ext]
    
    ss_dict = {'x1': data[1], 'x2': data[2], 
              'x3': data[3], 'x4': data[4], 'x5': data[5]}
    return par_dict, ss_dict, data

par_dict, ss_dict, data = get_data(PC, curve = 'EQrisk2', special_point = 'BP1', par_dict = eq1_h1_par_dict)
par_dict


# In[243]:


def plot_time_perturbed_steady_state(PAR_dict = par_dict, ss_dict = ss_dict, data = data, tend = 10_000, ode = ode, par = 'risk', random_bool = True, eps = 0.1, par_dict = eq1_h1_par_dict):
    """
    function to plot time series of the data
    :param par_val: parameter used for simulated bifurcation
    :param ss_dict: dictionrary of steady states for bifurcation
    :param data: data obtained from get_data()
    :param tend: final time for plot
    :param ode: ode geneeated from generate_ode()
    :param par: string used for the par_val
    :param random_bool: boolean to use random initial conditions
    :param eps: small value used to perturb about the steady state
    :param par_dict: dictionary for parameters used for the bifurcation
    :return: plotted in time figures and pts -- pointset
    """
    key_vals = ['x1', 'x2', 'x3', 'x4', 'x5']
    ss_dict_perturbed = {}
    if random_bool:
        eps0 = eps
    else:
        eps0 = 0
            
    # generate new initial conditions!
    for k, v in ss_dict.items():
        sgn = np.sign(eps - 1/2)
        ss_dict_perturbed[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1)) # perturb new steady state
    # generate new system
    ode = generate_ode(PAR_dict, ss_dict_perturbed, tf = tend)    # generate a pointset
    pts = ode.compute('sample').sample(dt = 1)
    
    # plot the variables
    t = pts['t']
    sg = pts['x1']
    sb = pts['x2']
    ib = pts['x3']
    v = pts['x4']
    phi = pts['x5']
    
    # auxillary
    bad = sb + ib
    good = 1 - (sb + ib + v)
    infected = 1 - (sg + sb + v)
    healthy = sg + sb + v
    ig = 1 - (sb + sg + ib + v)
    
    # plottiing equations of state
    plt.plot(t, sb, 'r', label = r'$S_B$')
    plt.plot(t, ib, 'r--', label = r'$I_B$')
    plt.plot(t, sg, 'b', label = r'$S_G$')
    plt.plot(t, v, 'b--', label = '$V$')
    plt.plot(t, phi, 'm', label = '$\phi$')
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

    plt.plot(t, infected, 'r', label=r'$I$')
    plt.plot(t, healthy, 'b', label=r'$S + V$')
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
    
    plt.plot(t, bad, 'r', label = r'$B$')
    plt.plot(t, good, 'k', label = r'$G$')
    plt.plot(t, v, 'b', label = r'$V$')
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()    
    
    return pts


# In[244]:


def generate_bifurcation_2d(PC, curve = 'EQ1', special_point = 'H2', xpar = 'misinformation', ypar = 'education', max_n_points = 80, curve_type = 'H-C1', par_dict = eq2_h2_par, name_curve = 'HO2'):
    """
    funciton to generate a co-dimension two bifurcation given an initial starting point on the curve
    :param PC: PyCont generated from generate_bifurcation
    :param curve: curve created from the generate_bifurcation plot
    :param special_point: LP, HO, BP, B, etc. special points determined from the user input for the previous curve
    :param xpar: parameter for the x value (original parameter used for the codimension 1 curve bifurcation)
    :param ypar: parameter for the y value (extension from the codimension 1 bifurcation previously)
    :param max_n_points: maximum number of points
    :param curve_type: Use: H-C1, H-C2, LP-C
    :param par_dict: dictionary of parameters used for the initialization of continuation
    :param name_curve: name of the curve that will be generated for the codimension two bifurcation!
    :return: plotted bifurcation via matplotlib
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
        xlab = r'$\chi_{bb}$'
    elif xpar == 'protection':
        xlab = '$\delta$'
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
        ylab = r'$\chi_{bb}$'
    elif ypar == 'protection':
        ylab = '$\delta$'
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
    
    # grab the data from the user supplied speical point
    point_special = PC[curve].getSpecialPoint(special_point)

    # get the steady state values associated to the special point!
    ss_dict = {'x1': point_special['x1'],
               'x2': point_special['x2'],
               'x3': point_special['x3'],
               'x4': point_special['x4'],
               'x5': point_special['x5']} 
    
    par_dict[xpar] = point_special[xpar] # assign the parameter value at the dictionary value of the special point
    
    # now we generate a continuation!
    PCargs = dst.args(name=name_curve, type=curve_type)
    PCargs.freepars = [xpar, ypar]  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 1e-4
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations / hopf bifurcations!
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B' # boundary for the continuation
    # generate a numerical continuation
    PC.newCurve(PCargs)

    # continue backwards and forwards!
    PC[name_curve].forward()
    PC[name_curve].backward()

    
    # now we display the continuation curve with respect to the variables!
    PC.display([xpar, ypar])

    # display the bifurcation!
    PC.display([xpar, ypar], stability=True)  # x variable vs y variable!
    # disable the boundary
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
    plt.title('')  # no title
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.show()

    PC_2d = PC

    return PC_2d


# In[246]:


pts = plot_time_perturbed_steady_state(PAR_dict = par_dict, ss_dict = ss_dict, data = data, tend = 10_000, ode = ode, par = 'risk', random_bool = True, eps = 0.1)


# In[253]:


plot_nullclines_new(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xlow = -0.0, xhigh = 0.65, n_bin = 200)
plot_nullclines_new(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200)
plot_nullclines_new(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65, n_bin = 200)


# In[ ]:


# getting data
par_dict, ss_dict, data = get_data(PC, curve = 'EQrisk2', special_point = 'BP1', par_dict = eq1_h1_par_dict)

# using bifurcation data to get perturbed pointset!
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict, ss_dict = ss_dict, data = data, tend = 10_000, ode = ode, par = 'risk', random_bool = True, eps = 0.2)

# plotting nullclines
plot_nullclines_new(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)


# In[233]:


par_dict['protection'] = 0.90


# In[203]:


PC_2d_risk = generate_bifurcation_2d(PC, curve = 'EQrisk2', special_point = 'BP1', xpar = 'risk', ypar = 'protection', par_dict = par_dict, name_curve = 'H6', max_n_points=30, curve_type = 'H-C1')


# In[204]:


PC_2d_risk_education = generate_bifurcation_2d(PC, curve = 'EQrisk2', special_point = 'BP1', xpar = 'risk', ypar = 'education', par_dict = par_dict, name_curve = 'H1', max_n_points=50, curve_type = 'H-C1')


# In[212]:


PC_2d_risk_education = generate_bifurcation_2d(PC, curve = 'EQrisk', special_point = 'H1', xpar = 'risk', ypar = 'education', par_dict = par_dict, name_curve = 'new', max_n_points=50, curve_type = 'H-C1')


# In[209]:





# In[227]:


PC_2d_risk_education['new'].info()
PC_2d_risk_education['new'].getSpecialPoint('DH2')


# In[235]:


# getting data
#PC_2d_risk_education['new'].info()
"""Special Points
-------------- 

P1, P2, GH1, DH2, DH1"""
par_dict_new, ss_dict_new, data_new = get_data(PC_2d_risk_education, curve = 'new', special_point = 'DH1', par_dict = par_dict, par = 'risk', par_ext = 'education')

print(par_dict, ss_dict)
# using bifurcation data to get perturbed pointset!
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict_new, ss_dict = ss_dict_new, data = data_new, tend = 1_000, ode = ode, random_bool = True, eps = 0.1)

# plotting nullclines
plot_nullclines_new(option = 'A', PTS = pts, par_dict = par_dict_new, ss_dict = ss_dict_new, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'B', PTS = pts, par_dict = par_dict_new, ss_dict = ss_dict_new, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'C', PTS = pts, par_dict = par_dict_new, ss_dict = ss_dict_new, evecs_bool = False, xhigh = 0.65)


# In[328]:


def generate_protection_bifurcation(ODE = ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 150, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    ode = generate_ode(par_dict, ics_dict, tf = tend)    # generate a pointset
    #pts, ss_dict = generate_pointset(ode)
    
    # use the initial conditions at the steady state
    #ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQprotection', type = 'EP-C')
    PCargs.freepars = ['protection']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 1e-3
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'
    # generate a numerical continuation
    PC.newCurve(PCargs)

    # continue backwards and forwards!
    PC['EQprotection'].forward()
    PC['EQprotection'].backward()

    
    PCargs.name = 'EQprotection2'
    PCargs.type = 'EP-C'
    PCargs.initpoint = {'x1':  0.02179992969509327,
                    'x2':  0.21186033176092678,
                    'x3':  0.07439263859553721,
                    'x4':  0.6784593698650656,
                    'x5':  0.011113221022650486,
                       'protection': 0.5}
    
    PCargs.freepars = ['protection']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.0*max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    #PC.update(PCargs)
    PC.newCurve(PCargs)
    #PC['EQprotection2'].forward()
    #PC['EQprotection2'].backward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$']
    col_array = ['b', 'r', 'k', 'orange']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['protection', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\delta$')
        plt.show()
        
    return PC


# In[329]:


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

eq1_risk_bp1_ss = {
'x1':  0.02179992969509327,
'x2':  0.21186033176092678,
'x3':  0.07439263859553721,
'x4':  0.6784593698650656,
'x5':  0.011113221022650485}

eq1_risk_bp1_par_dict = {'risk':  0.12952930209570576, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

par_dict = eq1_risk_lp1_par_dict
ss_dict = eq1_risk_lp1_ss


# In[331]:


# generate a bifurcation!
PC = generate_protection_bifurcation(ics_dict = ss_dict, par_dict = par_dict, tend = 300, max_points = 120)

# getting data
par_dict, ss_dict, data = get_data(PC, curve = 'EQprotection', special_point = 'LP1', par_dict = par_dict, par = 'protection')

# using bifurcation data to get perturbed pointset!
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict, ss_dict = ss_dict, data = data, tend = 10_000, ode = ode, par = 'protection', random_bool = True, eps = 0.1)

# plotting nullclines
plot_nullclines_new(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)
plot_nullclines_new(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.65)


# In[ ]:




