import matplotlib.pyplot as plt
import generate_ode
import PyDSTool as dst
from PyDSTool import *

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


def generate_protection_bifurcation(ODE=ode, ics_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, max_points=150, tend=1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    ode = generate_ode(par_dict, ics_dict, tf=tend)  # generate a pointset
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name='EQprotection', type='EP-C')
    PCargs.freepars = ['protection']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.001
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
    PC['EQprotection'].forward()
    PC['EQprotection'].backward()

    PCargs.name = 'EQprotection2'
    PCargs.type = 'EP-C'

    PCargs.initpoint = {'x1': 0.00045686505623685283,
                        'x2': 0.1894318195395322,
                        'x3': 0.1968176793050778,
                        'x4': 0.6064537764311808,
                        'x5': 0.9999999999999999,
                        'protection': 0.92}

    PCargs.freepars = ['protection']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.2 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.001
    PCargs.MinStepSize = 1e-7
    PCargs.StepSize = 1e-5
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQprotection2'].forward()
    PC['EQprotection2'].backward()

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