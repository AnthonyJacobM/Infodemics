import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt

path = r'D:\Users\antho\PycharmProjects\Infodemics\figures'


# get all dictionary for parameters and initial conditions!

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

def generate_bifurcation_2d(PC, curve='EQ1', special_point='H2', xpar='misinformation', ypar='education',
                            max_n_points=80, curve_type='H-C2', par_dict=eq2_h2_par, name_curve='HO2'):
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
    elif xpar == 'recovery':
        xlab = r'$\gamma$'
    elif xpar == 'ace':
        xlab = '$a$'
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
    elif ypar == 'recovery':
        ylab = r'$\gamma$'
    elif ypar == 'ace':
        ylab = '$a$'
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

    par_dict[xpar] = point_special[xpar]  # assign the parameter value at the dictionary value of the special point

    # now we generate a continuation!
    PCargs = dst.args(name=name_curve, type=curve_type)
    PCargs.freepars = [xpar, ypar]  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_n_points  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.1
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 1e-3
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations / hopf bifurcations!
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2
    PCargs.StopAtPoints = 'B'  # boundary for the continuation
    # generate a numerical continuation
    PC.newCurve(PCargs)

    # continue backwards and forwards!
    PC[name_curve].forward()

    # start at the initial point and generate a backwards continuation!
    PCargs.initpoint = point_special
    PCargs.StepSize = 1e-2
    PC.update(PCargs)
    PC[name_curve].backward()

    # now we display the continuation curve with respect to the variables!
    PC.display([xpar, ypar])

    # display the bifurcation!
    PC.display([xpar, ypar], stability=True)  # x variable vs y variable!
    # disable the boundary
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
    PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')

    # get the data obtained in the figure!
    lc_x = PC[name_curve].sol[xpar]
    lc_y = PC[name_curve].sol[ypar]

    # determine the limit cycle boundaries in x and y
    print(np.size(lc_x))
    print(np.size(lc_y))

    plt.fill(lc_x, lc_y)
    plt.title('')  # no title
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.xlim(np.min(lc_x) * 0.95, np.max(lc_x) * 1.05)
    plt.ylim(np.min(lc_y) * 0.95, np.max(lc_y) * 1.05)
    plt.show()

    PC_2d = PC

    return PC_2d

