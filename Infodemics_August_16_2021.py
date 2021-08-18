#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


# In[2]:


# define generation of ode's
import PyDSTool as dst
from PyDSTool import *

# intialized parameters and initial conditions!
tend = 10_000
# recovery = \gamma, belief = m, risk = r, protection = \delta, education = \tilde{\chi}_{gb}, misinformation = \tilde{\chi}_{bg}
# infection_good = \chi_{bg}, infection_bad = \chi_{bb}

# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

# parameters for branching point at risk bifurcation!
par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}
eq1_h1_par_dict = par_bp_r
eq1_h1_par_dict['risk'] = 1.635295791362042
eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927

def generate_ode(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = tend, version_rev_bool = False):
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
    ace = Par(par_dict['ace'], 'ace')

    DSargs.pars = [recovery, belief, risk, protection, education,
                   misinformation, infection_good, infection_bad, ace]

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
                      'protection': [0, 1], 'risk': [0, 6],
                      'belief': [0, 1], 'recovery': [0, 1],
                      'ace': [0, 5]}

    # generate bounds on state variables!
    DSargs.xdomain = {'x1': [0, 1], 'x2': [0, 1],
                      'x3': [0, 1], 'x4': [0, 1],
                      'x5': [0, 1]}

    # generate right hand side of the differential equations!
    if version_rev_bool == False:
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (x5 + (infection_good + misinformation) * x3 + misinformation * x2)
        x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (1 - protection) * infection_good * x4)
        x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
        x5rhs = belief * x5 * (1 - x5) * (ace * (1 - x2 - x3 - x4) + (1 - x1 - x2 - x4) - risk * x4)
    else:
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (x5 + (infection_good + misinformation) * x3 + misinformation * x2)
        x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (1 - protection) * infection_good * x4)
        x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
        x5rhs = belief * x5 * (1 - x5) * (ace * (1 - x2 - x3 - x4) + (1 - x1 - x2 - x4) - risk * x4)

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


# In[3]:


# initialize the default settings for plotting in matplotlib!
# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0


# Master code to run for all the bifurcations
# set up an initialization for parameters ans initial conditions


# In[4]:


# initialize parameter and initial condition dictionaries!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

# sample dictionaries to start at!
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


tend = 10_000
# generate a function for the system
def gen_sys(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = tend):
    """
    function to generate a system used for integration
    :param par_dict: dictionary for parameters
    :param ics_dict: dicitonary for initial conditions
    :param tf: final time
    :return: the solution
    """

    # unpack the dictionaries
    recovery = par_dict['recovery']
    belief = par_dict['belief']
    infection_good = par_dict['infection_good']
    infection_bad = par_dict['infection_bad']
    misinformation = par_dict['misinformation']
    education = par_dict['education']
    risk = par_dict['risk']
    protection = par_dict['protection']
    ace = par_dict['ace']

    # determine initial conditions
    x0 = np.array([ics_dict['x1'], ics_dict['x2'], ics_dict['x3'], ics_dict['x4'], ics_dict['x5']])

    # generate a time sequence!
    t = np.linspace(0, tf, tf + 1)

    # generate a function for the system
    def sys(X, t = 0):
        """
        function for determining the ode rhs
        :param X: array
        :param t: time real number
        :return: Y integrated solution
        """
        Y = np.zeros(len(X)) # create empty bin!

        Y[0] = recovery * (1 - X[0] - X[1] - X[2] - X[3]) - X[0] * (X[4] + (infection_good + misinformation) * X[2] + misinformation * X[1])
        Y[1] = misinformation * X[0] * (X[1] + X[2]) - X[2] * (infection_bad * X[1] - recovery)
        Y[2] = X[2] * (infection_bad * X[1] - recovery - education * (X[0] + (1 - X[0] - X[1] - X[2] - X[3])) + (1 - protection) * infection_good * X[3])
        Y[3] = X[4] * X[0] - (1 - protection) * infection_good * X[3] * X[2]
        Y[4] = belief * X[4] * (1 - X[4]) * (ace * (1 - X[1] - X[2] - X[3]) + X[2] + (1 - X[0] - X[1] - X[2] - X[3]) - risk * X[3])

        return Y

    # integrate the solution
    z, infodict = odeint(sys, x0, t, full_output = True)

    return z, t






def example():
    """
    function to test
    :return:
    """
    eq1_h1_ss_dict = {'x1': 0.1652553343953094,
                      'x2': 0.4608116686366218,
                      'x3': 0.09068387295130048,
                      'x4': 0.14189412748039304,
                      'x5': 0.0003737491655869812}

    eq1_h1_par_dict = par_dict_def
    eq1_h1_par_dict['risk'] = 1.635295791362042

    eq1_bp1_par_dict = par_dict_def
    eq1_bp1_par_dict['risk'] = 0.34021985318961456

    eq1_bp1_ss_dict = {'x1': 0.00057159125563,
                       'x2': 0.189492234608487,
                       'x3': 0.197046892184315,
                       'x4': 0.6043308284126,
                       'x5': 1.0}

    eq1_dh2_ss_dict = {
    'x1': 0.17776272966154416,
    'x2': 0.3292021937121564,
    'x3': 0.1719729462024596,
    'x4': 0.17081347653498602,
    'x5': 0.0007932001548364984
    }

    eq1_dh2_par_dict = par_dict_def
    eq1_dh2_par_dict['risk'] = 1.8863944849535226
    eq1_dh2_par_dict['education'] = 0.16043564034659807

    eq_ss_eps_dict = {}

    ss_dict_gh = {'x1':  0.3715711793910016,
                  'x2':  0.5267668444942307,
                  'x3':  0.004021761741446761,
                  'x4':  0.0894146579431335,
                  'x5':  4.645412394314906e-06}

    par_dict_gh = par_dict_def
    par_dict_gh['risk'] = 0.13697215259072273
    par_dict_gh['misinformation'] = 0.0025469952913920665

    ss_dict_dh = {'x1': 0.04590829474163592,
               'x2': 0.2937449920912099,
               'x3': 0.12666881948434064,
               'x4': 0.4557281813211489,
               'x5': 0.006035672748913323}
    par_dict_dh = par_dict_def
    par_dict_dh['risk'] = 0.44899249208773284
    par_dict_dh['misinformation'] = 0.2538934237606571

    ss_dict = eq1_dh2_ss_dict
    par_dict = eq1_dh2_par_dict

    eps0 = 0.00001
    for k, v in ss_dict.items():
        print(k, v)
        sgn = np.sign(np.random.rand(1) - 1 / 2)[0]
        ss_dict[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1))[0]  # perturb new steady state

    # example
    z, t = gen_sys(par_dict=par_dict, ics_dict=ss_dict)

    sg, sb, ib, v, phi = z.T
    plt.plot(sg, sb)
    plt.show()

    plt.plot(t, sg, 'b', label = r'$S_G$')
    plt.plot(t, sb, 'r', label = r'$S_B$')
    plt.plot(t, ib, 'r--', label = r'$I_B$')
    plt.plot(t, v, 'b--', label = r'$V$')
    plt.plot(t, phi, 'm', label = '$\phi$')
    plt.legend()
    plt.xlabel('t (Days)')
    plt.show()


# In[36]:


import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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


# this is the steady state value for misinformation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927


def generate_a_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 200, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    #ode = generate_ode(par_dict, ics_dict, tf = tend)    # generate a pointset
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQace', type = 'EP-C')
    PCargs.freepars = ['ace']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.001
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
    #PC['EQa'].forward()
    PC['EQace'].backward()


    PCargs.name = 'EQanew'
    PCargs.type = 'EP-C'

    # PCargs.initpoint = {'x1': 0.00045686505623685283,
    #                    'x2': 0.1894318195395322,
    #                    'x3': 0.1968176793050778,
    #                    'x4': 0.6064537764311808,
    #                    'x5': 0.9999999999999999,
    #                    'misinformation': 0.35}


    """
    LP Point found 
    ========================== 
    0 : 
    x1  =  0.05291675637403153
    x2  =  0.20334289508621275
    x3  =  0.40825986806257114
    x4  =  0.2822227208716318
    x5  =  0.010451453370836366
    education  =  0.06208213130305199
    """


    dict = {'x1': 0.05291675637403153,
            'x2': 0.20334289508621275,
            'x3': 0.40825986806257114,
            'x4': 0.2822227208716318,
            'x5': 0.010451453370836366,
            'education': 0.0}


    dict = {'x1': 5.959501440455008e-14,
            'x2': 0.1891891891891843,
            'x3': 0.8108108108108129,
            'x4': -5.434278100754251e-14,
            'x5': 1.0,
            'education': 0.0}



    PCargs.initpoint = dict

    PCargs.freepars = ['ace']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.2 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-7
    PCargs.StepSize = 1e-4
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    #PC['EQanew'].forward()
    #PC['EQanew'].backward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['ace', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        #PC.plot.toggleLabels(visible='off', bylabel=None, byname='H4')

        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$a$')
        file_name = f"\Ace_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[37]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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


# this is the steady state value for misinformation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
#eq2_h2_par['misinformation'] = 0.0660611192767927


def generate_a_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 200, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    #ode = generate_ode(par_dict, ics_dict, tf = tend)    # generate a pointset
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQace', type = 'EP-C')
    PCargs.freepars = ['ace']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.001
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
    #PC['EQa'].forward()
    PC['EQace'].backward()


    PCargs.name = 'EQanew'
    PCargs.type = 'EP-C'

    # PCargs.initpoint = {'x1': 0.00045686505623685283,
    #                    'x2': 0.1894318195395322,
    #                    'x3': 0.1968176793050778,
    #                    'x4': 0.6064537764311808,
    #                    'x5': 0.9999999999999999,
    #                    'misinformation': 0.35}


    """
    LP Point found 
    ========================== 
    0 : 
    x1  =  0.05291675637403153
    x2  =  0.20334289508621275
    x3  =  0.40825986806257114
    x4  =  0.2822227208716318
    x5  =  0.010451453370836366
    education  =  0.06208213130305199
    """


    dict = {'x1': 0.05291675637403153,
            'x2': 0.20334289508621275,
            'x3': 0.40825986806257114,
            'x4': 0.2822227208716318,
            'x5': 0.010451453370836366,
            'education': 0.0}


    dict = {'x1': 5.959501440455008e-14,
            'x2': 0.1891891891891843,
            'x3': 0.8108108108108129,
            'x4': -5.434278100754251e-14,
            'x5': 1.0,
            'education': 0.0}



    PCargs.initpoint = dict

    PCargs.freepars = ['ace']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.2 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-7
    PCargs.StepSize = 1e-4
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    #PC['EQanew'].forward()
    #PC['EQanew'].backward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['ace', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        #PC.plot.toggleLabels(visible='off', bylabel=None, byname='H4')

        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$a$')
        file_name = f"\Ace_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[39]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


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


# this is the steady state value for misinformation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927


def generate_education_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 200, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    #ode = generate_ode(par_dict, ics_dict, tf = tend)    # generate a pointset
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQeducation', type = 'EP-C')
    PCargs.freepars = ['education']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.001
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
    PC['EQeducation'].forward()
    PC['EQeducation'].backward()


    PCargs.name = 'EQeducation2'
    PCargs.type = 'EP-C'

    # PCargs.initpoint = {'x1': 0.00045686505623685283,
    #                    'x2': 0.1894318195395322,
    #                    'x3': 0.1968176793050778,
    #                    'x4': 0.6064537764311808,
    #                    'x5': 0.9999999999999999,
    #                    'misinformation': 0.35}


    """
    LP Point found 
    ========================== 
    0 : 
    x1  =  0.05291675637403153
    x2  =  0.20334289508621275
    x3  =  0.40825986806257114
    x4  =  0.2822227208716318
    x5  =  0.010451453370836366
    education  =  0.06208213130305199
    """


    dict = {'x1': 0.05291675637403153,
            'x2': 0.20334289508621275,
            'x3': 0.40825986806257114,
            'x4': 0.2822227208716318,
            'x5': 0.010451453370836366,
            'education': 0.0}


    dict = {'x1': 5.959501440455008e-14,
            'x2': 0.1891891891891843,
            'x3': 0.8108108108108129,
            'x4': -5.434278100754251e-14,
            'x5': 1.0,
            'education': 0.0}



    PCargs.initpoint = dict

    PCargs.freepars = ['education']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.2 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-7
    PCargs.StepSize = 1e-4
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    #PC['EQeducation2'].forward()
    #PC['EQeducation2'].backward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k'] 
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['education', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        #PC.plot.toggleLabels(visible='off', bylabel=None, byname='H4')

        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\epsilon$')
        file_name = f"\education_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[9]:


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
        xlab = r'$\epsilon$'
    elif xpar == 'misinformation':
        xlab = r'$\mu$'
    elif xpar == 'infection_good':
        xlab = r'$\chi$'
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
        ylab = r'$\epsilon$'
    elif ypar == 'misinformation':
        ylab = r'$\mu$'
    elif ypar == 'infection_good':
        ylab = r'$\chi$'
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
    PCargs.MaxStepSize = 0.10
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
    plt.xlim(np.min(lc_x) * 0.85, np.max(lc_x) * 1.15)
    plt.ylim(np.min(lc_y) * 0.85, np.max(lc_y) * 1.15)
    plt.show()

    PC_2d = PC

    return PC_2d


# In[10]:


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

def generate_education_lc_boundary(execution = 'generate_lc'):
    """
    :param execution: use generate_lc, generate_hm, plot_hm
    function to generate the risk boundary of the limit cycles
    :return: plotted matplotlib fiugre
    """
    # generate a risk bifurcation using the user supplied ics and dictionary of parameters
    PC_education, par_dict_education, ss_dict_education, data_education = plot_education_bifurcation(par_dict=eq1_h1_par_dict,
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

    max_step_size_bin = [100, 100, 85]
    aux_par_list = ['infection_good', 'infection_bad', 'protection']
    aux_par_list = ['protection']
    # iterate over the array and determine the limit cycle boundary
    for p, p0 in enumerate(aux_par_list):
        if execution == 'generate_lc':
            generate_limit_cycle_boundary(PC_education, curve='EQeducation',
                                          special_point='H1', xpar='education',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          max_n_points=max_step_size_bin[p],
                                          curve_type='H-C1', name_curve='Heducation' + str(p),
                                          save_bool=True)
        elif execution == 'generate_hm':
            generate_heat_map_limit_cycle(PC_education, curve='EQeducation',
                                          special_point='H1', xpar='education',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          ss_dict=eq1_h1_ss, max_n_points=max_step_size_bin[p], load_bool = True,
                                          curve_type='H-C1',  name_curve = 'Heducation_new' + str(p), n_bin=100, tend=5_000, eps=0.0005)
        else:
            file_vac = f"education_{p0}_vac.txt"
            file_bad = f"education_{p0}_bad.txt"
            file_inf = f"education_{p0}_inf.txt"
            file_xlimit = f"lc_education_{p0}_xlimit.txt"
            file_ylimit = f"lc_education_{p0}_ylimit.txt"

            data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
            data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
            data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)
            xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
            ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

            plot_lc_heatmap(data=data_bad, zvar='inf', xpar='education', ypar=p0, cmap='Reds',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_bad, zvar='bad', xpar='education', ypar=p0, cmap='Greys',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_vac, zvar='vac', xpar='education', ypar=p0, cmap='Blues',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])

#generate_education_lc_boundary(execution = 'generate_lc')
#generate_education_lc_boundary(execution = 'generate_hm')
#generate_education_lc_boundary(execution = 'plot_hm')


# In[11]:


import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

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
        xlab = r'$\epsilon$'
    elif xpar == 'misinformation':
        xlab = r'$\mu$'
    elif xpar == 'infection_good':
        xlab = r'$\chi$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}$'
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
        ylab = r'$\epsilon$'
    elif ypar == 'misinformation':
        ylab = r'$\mu$'
    elif ypar == 'infection_good':
        ylab = r'$\chi$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}$'
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

#array = np.random.rand(5)
#np.savetxt('test.txt', array, delimiter = ',')


# In[40]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

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

g
# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927

def generate_infection_bad_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 250, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQinfection_bad', type = 'EP-C')
    PCargs.freepars = ['infection_bad']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.005
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
    PC['EQinfection_bad'].forward()
    PC['EQinfection_bad'].backward()

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

    """PCargs.name = 'EQrisk2'
    PCargs.type = 'EP-C'
    PCargs.initpoint = {'x1': 0.0005714445346933421,
                        'x2': 0.1893893682498801,
                        'x3': 0.1968691906600376,
                        'x4': 0.6047210552785887,
                        'x5': 1.0,
                        'risk': 0.0}

    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.0 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQrisk2'].forward()"""

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['infection_bad', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\hat{\chi}$')
        file_name = f"\Infection_bad_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[41]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

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

def generate_infection_good_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 250, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQinfection_good', type = 'EP-C')
    PCargs.freepars = ['infection_good']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.005
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
    PC['EQinfection_good'].forward()
    PC['EQinfection_good'].backward()

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

    """PCargs.name = 'EQrisk2'
    PCargs.type = 'EP-C'
    PCargs.initpoint = {'x1': 0.0005714445346933421,
                        'x2': 0.1893893682498801,
                        'x3': 0.1968691906600376,
                        'x4': 0.6047210552785887,
                        'x5': 1.0,
                        'risk': 0.0}

    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.0 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQrisk2'].forward()"""

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['infection_good', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\chi_{bg}$')
        file_name = f"\Infection_good_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[13]:


import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt

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

def generate_limit_cycle_boundary(PC, curve = 'EQmisinformation', special_point = 'H1', xpar = 'misinformation', ypar = 'education', par_dict = eq2_h2_par_dict, max_n_points = 35, curve_type = 'H-C1', name_curve = 'HO1', xmax = '', ymax = '', par_x_lb = '', par_x_ub = '', par_y_lb = '', par_y_ub = '', save_bool = False):
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
    :param par_x_lb: lower bound on the parameter on x
    :param par_x_ub: upper bound on the parameter on x
    :param par_y_lb: lower bound on the parameter on y
    :param par_y_ub: upper bound on the parameter on y
    :return: plotted figure in matplotlib
    """


    if xpar == 'risk':
        xlab = '$r$'
    elif xpar == 'education':
        xlab = r'$\epsilon$'
    elif xpar == 'misinformation':
        xlab = r'$\mu$'
    elif xpar == 'infection_good':
        xlab = r'$\chi$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}$'
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
        ylab = r'$\epsilon$'
    elif ypar == 'misinformation':
        ylab = r'$\mu$'
    elif ypar == 'infection_good':
        ylab = r'$\chi$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}$'
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
    # get the current axes limits
    xlimit = np.array(ax.get_xlim())
    ylimit = np.array(ax.get_ylim())
    file_name = f"\lc_{xpar}_{ypar}.jpeg"
    plt.savefig(path + file_name, dpi=300)
    plt.show()

    # generate a plot for the limit cycle boundary!
    plt.plot(x1_ss, x2_ss, 'b', lw=5)
    plt.fill(x1_ss, x2_ss, 'k')
    plt.xlabel(r'$S_G$', fontsize=18)
    plt.ylabel(r'$S_B$', fontsize=18)
    # get the current axes limits
    xlimit = np.array(ax.get_xlim())
    ylimit = np.array(ax.get_ylim())
    plt.show()

    # generate a plot for the limit cycle boundary!
    plt.plot(x3_ss, x4_ss, 'b', lw=5)
    plt.fill(x3_ss, x4_ss, 'k')
    plt.xlabel(r'$I_B$', fontsize=18)
    plt.ylabel(r'$V$', fontsize=18)
    # get the current axes limits
    xlimit = np.array(ax.get_xlim())
    ylimit = np.array(ax.get_ylim())
    plt.show()

    if par_x_lb == '':
        par_x_lb = xlimit[0]
    if par_x_ub == '':
        par_x_ub = xlimit[-1]
    if par_y_lb == '':
        par_y_lb = ylimit[0]
    if par_y_ub == '':
        par_y_ub = ylimit[-1]

    if par_x_lb == '' and par_x_ub == '':
        xlimit = np.array([par_x_lb * 1.05, 0.95 * par_x_ub])
    else:
        xlimit = np.linspace(par_x_lb * 1.05, 0.95 * par_x_ub, 100)
    if par_y_lb == '' and par_y_ub == '':
        ylimit = np.array([par_y_lb * 1.05, 0.95 * par_y_ub])
    else:
        ylimit = np.linspace(par_y_lb * 1.05, 0.95 * par_y_ub, 100)

    # get axes limit
    # determine the axes limits for the figure via user input
    """x_lb = input('par_x_lb')
    x_ub = input('par_x_ub')
    y_lb = input('par_y_lb')
    y_ub = input('par_y_ub')
    xlimit = [x_lb, x_ub]
    ylimit = [y_lb, y_ub]
    print('xlimit: ', xlimit, 'ylimit: ', ylimit)"""

    if save_bool == True:
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


# In[42]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


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

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927

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




# --
def generate_misinformation_bifurcation(ode, ics_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, max_points=200,
                                        tend=1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ODE: generated previously
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    # ode = generate_ode(par_dict, ics_dict, tf=tend)  # generate a pointset
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name='EQmisinformation', type='EP-C')
    PCargs.freepars = ['misinformation']  # should be one of the parameters from DSargs.pars --

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
    PC['EQmisinformation'].forward()
    PC['EQmisinformation'].backward()

    PCargs.name = 'EQmisinformation2'
    PCargs.type = 'EP-C'

    # PCargs.initpoint = {'x1': 0.00045686505623685283,
    #                    'x2': 0.1894318195395322,
    #                    'x3': 0.1968176793050778,
    #                    'x4': 0.6064537764311808,
    #                    'x5': 0.9999999999999999,
    #                    'misinformation': 0.35}

    dict = {'x1': 0.0005739017705169474,
            'x2': 0.18964617945844123,
            'x3': 0.19705914738387598,
            'x4': 0.6039697520726657,
            'x5': 0.9999999999999999,
            'misinformation': 0.15}

    PCargs.initpoint = dict

    PCargs.freepars = ['misinformation']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.2 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.005
    PCargs.MinStepSize = 1e-7
    PCargs.StepSize = 1e-4
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    # PC['EQmisinformation2'].forward()
    # PC['EQmisinformation2'].backward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['misinformation', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\mu$')
        file_name = f"\misinformation_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[15]:


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
#generate_misinformation_lc_boundary(execution = 'plot_hm')


# In[16]:


from PyDSTool import *
import PyDSTool as dst

def generate_pointset(ode, save_bool = False, save_ss_data_name = 'infodemics_default_ss_data.txt', save_pts_data_name = 'infodemics_default_pts_data.txt'):
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


# In[43]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


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


def generate_protection_bifurcation(ode, ics_dict=eq1_h1_ss, par_dict=eq1_h1_par_dict, max_points=150, tend=1000):
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
    #ode.set(ics = ss_dict)

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
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['protection', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\delta$')
        file_name = f"\protection_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[44]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

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

def generate_recovery_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 250, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQrecovery', type = 'EP-C')
    PCargs.freepars = ['recovery']  # should be one of the parameters from DSargs.pars --

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
    PC['EQrecovery'].forward()
    PC['EQrecovery'].backward()

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

    """PCargs.name = 'EQrisk2'
    PCargs.type = 'EP-C'
    PCargs.initpoint = {'x1': 0.0005714445346933421,
                        'x2': 0.1893893682498801,
                        'x3': 0.1968691906600376,
                        'x4': 0.6047210552785887,
                        'x5': 1.0,
                        'risk': 0.0}

    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars
    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = int(1.0 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQrisk2'].forward()"""

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x55']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['recovery', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$\gamma$')
        file_name = f"\Recovery_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[45]:


import matplotlib.pyplot as plt
import PyDSTool as dst
from PyDSTool import *

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

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

def generate_risk_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 250, tend = 1000):
    """
    function to generate a bifurcation for risk of vaccination relative to infection
    :param ics_dict: dictionary of initial conditions
    :param par_dict: dictionary of parameters
    :param max_points: maximum number of points for a bifurcation
    :param tend: final point in the time domain
    :return: plotted bifurcation
    """
    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # generate a coninuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name = 'EQrisk', type = 'EP-C')
    PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = 0.005
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
    PCargs.MaxNumPoints = int(1.0 * max_points)  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 0.01
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 0.1e-3
    PCargs.StopAtPoints = 'B'
    PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

    # PC.update(PCargs)
    PC.newCurve(PCargs)
    PC['EQrisk2'].forward()

    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display(['risk', yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(r'$r$')
        file_name = f"\Risk_{z0}.jpeg"
        plt.savefig(path + file_name, dpi = 300)
        plt.show()

    return PC


# In[20]:


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

def generate_risk_lc_boundary(execution = 'generate_lc'):
    """
    :param execution: use generate_lc, generate_hm, plot_hm
    function to geenrate the risk boundary of the limit cycles
    :return: plotted matplotlib fiugre
    """
    # generate a risk bifurcation using the user supplied ics and dictionary of parameters
    PC_risk, par_dict_risk, ss_dict_risk, data_risk = plot_risk_bifurcation(par_dict=eq1_h1_par_dict,
                                                                            ics_dict=eq1_h1_ss,
                                                                           special_point='H1', tend=0,
                                                                            eps0=0.00001)
    # use auxillary parameters
    aux_par_list = ['misinformation', 'education', 'recovery', 'infection_bad']
    max_step_size_bin = [100, 115, 115]
    x_lb_bin = [1.2, 1.2, 1.2, 1.2]
    x_ub_bin = [2.0, 2.0, 2.0, 2.0]
    y_lb_bin = [0.04, 0, 0.05, 0.1]
    y_ub_bin = [0.16, 0.8, 0.14, 0.85]

    aux_par_list = ['recovery']
    aux_par_list = ['misinformation', 'education']
    max_step_size_bin = [115]

    # iterate over the array and determine the limit cycle boundary
    for p, p0 in enumerate(aux_par_list):
        if execution == 'generate_lc':
            generate_limit_cycle_boundary(PC_risk, curve='EQrisk',
                                          special_point='H1', xpar='risk',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          max_n_points=max_step_size_bin[p],
                                          curve_type='H-C1', name_curve='Hrisk' + str(p),
                                          save_bool=True)
        elif execution == 'generate_hm':
            generate_heat_map_limit_cycle(PC_risk, curve='EQrisk',
                                          special_point='H1', xpar='risk',
                                          ypar=p0, par_dict=eq1_h1_par_dict,
                                          ss_dict=eq1_h1_ss, max_n_points=max_step_size_bin[p], load_bool = True,
                                          curve_type='H-C1',  name_curve = 'Hrisk_new' + str(p), n_bin=100, tend=5_000, eps=0.0005)
        else:
            file_vac = f"risk_{p0}_vac.txt"
            file_bad = f"risk_{p0}_bad.txt"
            file_inf = f"risk_{p0}_inf.txt"
            file_xlimit = f"lc_risk_{p0}_xlimit.txt"
            file_ylimit = f"lc_risk_{p0}_ylimit.txt"

            data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
            data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
            data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)
            xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
            ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

            plot_lc_heatmap(data=data_vac, zvar='inf', xpar='risk', ypar=p0, cmap='Reds',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_vac, zvar='bad', xpar='risk', ypar=p0, cmap='Greys',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])
            plot_lc_heatmap(data=data_vac, zvar='vac', xpar='risk', ypar=p0, cmap='Blues',
                            xlow=xlimit[0], xhigh=xlimit[-1], ylow=ylimit[0], yhigh=ylimit[-1])

#generate_risk_lc_boundary(execution = 'generate_lc')
#generate_risk_lc_boundary(execution = 'generate_hm')
#generate_risk_lc_boundary(execution = 'plot_hm')



# In[21]:


# dictionaries for parameters and initial conditions!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
            'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927


#ss_dict = ss_bp_r
#par_dict = par_hopf_r


def get_data(PC, curve='EQrisk', special_point='H1', par='risk', par_dict=eq1_h1_par_dict):
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

    ss_dict = {'x1': data[1], 'x2': data[2],
               'x3': data[3], 'x4': data[4], 'x5': data[5]}
    return par_dict, ss_dict, data


# In[22]:


import numpy as np
# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
            'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927

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
    a = par_dict['ace']

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
    x5rhs = belief * x5 * (1 - x5) * (a * (1 - x2 - x3 - x4) + (1 - x1 - x2 - x4) - risk * x4)

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


# In[23]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

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
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

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




# In[24]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

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

par_dict_default = eq1_h1_par_dict_new
ss_dict_default = eq1_h1_ss_new


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
    PC_education = generate_education_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_education, curve='EQeducation', special_point=special_point, par_dict=par_dict,
                                           par='education')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='education', random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.6,
                    n_bin=200, par='education')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.50, yhigh=0.5,
                    n_bin=200, par='education')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.75,
                    n_bin=200, par='education')

    return PC_education, par_dict, ss_dict, data




# In[25]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# change default mpl settings
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

# use one of the following for choice in colormaps!
cmap_bin = {'inferno': 'inferno', 'seismic': 'seismic', 'hot': 'hot', 'cool': 'cool', 'RdBu': 'RdBu',
            'plasma': 'plasma', 'viridis': 'viridis', 'YlGnBu': 'YlGnBu',
            'winter': 'winter', 'Blues': 'Blues', 'Reds': 'Reds', 'Greys': 'Greys'}

# default random data!
array = np.random.random((100, 100))


xlow = 1.0
xhigh = 1.65
ylow = 0.0
yhigh = 1.0


def plot_lc_heatmap(save_path = path, save_name = '\heatmap.jpg', data = array, zvar = 'inf', xpar = 'risk', ypar = 'misinformation', xlow = 1, xhigh = 2, ylow = 0, yhigh = 0.25, cmap = cmap_bin['inferno'], show_bool = True):
    """
    function to generte heatmaps with user input data
    :param save_path: path to save the figure
    :param save_name: file name unique to the figure
    :param data: user input numpy data
    :param xpar: parameter for the x in lc boundary
    :param ypar: parameter for the y in lc boundary
    :param xlow: parameter used for low value of x
    :param xhigh: parameter used for high value of x
    :param ylow: parameter used for low value of y
    :param yhigh: parameter used for high value of y
    :param cmap: color used for the heatmap; use one from the dictionary
    :param show_bool: boolean to show the figure
    :return: plotted heatmap!
    """
    # generate a new figure!
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)

    file_vac = f"{xpar}_{ypar}_vac.txt"
    file_bad = f"{xpar}_{ypar}_bad.txt"
    file_inf = f"{xpar}_{ypar}_inf.txt"
    file_xlimit = f"lc_{xpar}_{ypar}_xlimit.txt"
    file_ylimit = f"lc_{xpar}_{ypar}_ylimit.txt"

    data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
    data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
    data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)

    if zvar == 'vac':
        data = data_vac
        cbar_lab = 'Vaccinated'
        lc_col = 'r'
    elif zvar == 'inf':
        data = data_inf
        cbar_lab = 'Infected'
        lc_col = 'k'
    elif zvar == 'bad':
        data = data_bad
        cbar_lab = 'Bad'
        lc_col = 'b'
    else:
        data = data

    #xlimit = np.loadtxt(file_xlimit, delimiter=',', dtype=float)
    #ylimit = np.loadtxt(file_ylimit, delimiter=',', dtype=float)

    n = len(data)
    xlimit = [xlow, xhigh]
    ylimit = [ylow, yhigh]
    par_x_bin = np.linspace(xlimit[0], xlimit[-1], n)
    par_y_bin = np.linspace(ylimit[0], ylimit[-1], n)
    print(par_y_bin)
    row_label = [str(np.round(x0, 3)) for x, x0 in enumerate(par_x_bin)]
    col_label = [str(np.round(x0, 3)) for x, x0 in enumerate(par_y_bin)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if xpar == 'risk':
        xlab = '$r$'
    elif xpar == 'education':
        xlab = r'$\epsilon$'
    elif xpar == 'misinformation':
        xlab = r'$\mu$'
    elif xpar == 'infection_good':
        xlab = r'$\chi$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}$'
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
        ylab = r'$\epsilon$'
    elif ypar == 'misinformation':
        ylab = r'$\mu$'
    elif ypar == 'infection_good':
        ylab = r'$\chi$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}$'
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

    # generate spaces for col_label and row_label
    for z in range(0, len(row_label)):
        if z % 4 != 0:
            print(z % 4)
            row_label[z] = ''
            col_label[z] = ''



    df = pd.DataFrame(data, columns = col_label, index = row_label)
    s = sns.heatmap(df, cmap = cmap)
    #h_x = (xlimit[-1] - xlimit[0]) / 8
    #s.set_yticklabels(np.arange(xlimit[0], xlimit[-1], h_x))

    ax.set_xticklabels(col_label)
    ax.set_yticklabels(row_label)

    file_x = f"lc_{xpar}_{ypar}_xsol.txt"
    file_y = f"lc_{xpar}_{ypar}_ysol.txt"
    lc_x = np.loadtxt(file_x, delimiter=',', dtype=float)
    lc_y = np.loadtxt(file_y, delimiter=',', dtype=float)

    # get the axes for the linear spaced points
    xbin = ax.get_xticklabels()
    ybin = ax.get_yticklabels()

    xticks, xlabels = plt.xticks()
    yticks, ylabels = plt.yticks()

    # generate new xlabels and ylabels for the data depending on the lower and upper bounds of the parameters
    lb_x = xlimit[0]
    ub_x = xlimit[-1]
    lb_y = ylimit[0]
    ub_y = ylimit[-1]
    N_x = len(xlabels)
    N_y = len(ylabels)

    x_bin = np.linspace(lb_x, ub_x, N_x)
    y_bin = np.linspace(ub_y, lb_y, N_y)

    # generate new lables in a for loop
    xlabels_new = [str(np.round(x0,2)) for x, x0 in enumerate(x_bin)]
    ylabels_new = [str(np.round(y0,2)) for y, y0 in enumerate(y_bin)]

    # iterate using a for loop and replace the old labels
    for z, z0 in enumerate(xlabels):
        if z % 4 == 0:
            xlabels[z] = xlabels_new[z]
        else:
            xlabels[z] = ''

    for z, z0 in enumerate(ylabels):
        if z % 4 == 0:
            ylabels[z] = ylabels_new[z]
        else:
            ylabels[z] = ''



    # use the new labels and replace the old labels
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # normalize x and y array's!
    lc_x_norm = (lc_x - np.min(lc_x)) / (np.max(lc_x) - np.min(lc_x))
    lc_y_norm = (lc_y - np.min(lc_y)) / (np.max(lc_y) - np.min(lc_y))

    sns.lineplot(lc_x, lc_y, color = lc_col)
    #plt.fill(lc_x, lc_y, color = lc_col, alpha = 0.25)
    plt.show()


    # fill in between to obtain the limit cycle boundary
    # generate a plot for the limit cycle boundary!
    #sns.lineplot(x = lc_y_norm, y = lc_x_norm, color = 'b', lw=5)
    # plt.fill(lc_x, lc_y, 'k')
    # plt.xlabel(xlab, fontsize=18)
    # plt.ylabel(ylab, fontsize=18)
    # plt.xlim([0, xmax])
    # plt.ylim([0, ymax])
    # file_name = f"\lc_{xpar}_{ypar}.jpeg"
    # plt.savefig(path + file_name, dpi=300)
    # plt.show()
    ax.plot(lc_x, lc_y, color = 'b', lw = 2)
    min_data = np.min(data)
    max_data = np.max(data)
    v_min = np.max([min_data, 0])
    v_max = np.min([max_data, 1])
    im = ax.imshow(data, cmap = cmap,
                   extent = [xlimit[0], xlimit[-1], ylimit[0], ylimit[-1]])

    #plt.xlim(xlimit[0], xlimit[-1])
    #plt.ylim(ylimit[0], ylimit[-1])

    #plt.xlabel(xlab, fontsize = 18)
    #plt.ylabel(ylab, fontsize = 18)




    if show_bool == True:
        file_name = save_path + save_name
        plt.savefig(file_name, dpi = 300)
        plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize = (12,10), subplot_kw={'aspect': 'equal'})

    #ax1.set_yticks([int(j) for j in range(-4, 5)])
    #ax1.set_xticks([int(j) for j in range(-4, 5)])

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(15)
    for tick in ax1.get_xticklines() + ax1.get_yticklines():
        tick.set_markeredgewidth(2)
        tick.set_markersize(6)

    im = ax1.imshow(data, cmap=cmap, aspect = 0.10, vmin = v_min, vmax = v_max,
                    extent = [xlimit[0], xlimit[-1], ylimit[0], ylimit[-1]])
    ax1.set_xlim(xlimit[0], xlimit[-1])
    ax1.set_ylim(ylimit[0], ylimit[-1])

    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)

    #ax1.yaxis.set_tick_params(labelsize='xx-large')
    #ax1.xaxis.set_tick_params(labelsize='xx-large')


    divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="3.5%", pad=0.3)
    #cb = plt.colorbar(im, cax=cax)
    #cb.set_label(cbar_lab, fontsize = 18)
    ax1.plot(lc_x, lc_y, color = lc_col, lw = 5)
    #ax1.fill(lc_x, lc_y, color = lc_col, alpha = 0.25)
    plt.show()

    plt.close(fig)

    return lc_x, lc_y


# In[26]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# change default mpl settings
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

# use one of the following for choice in colormaps!
cmap_bin = {'inferno': 'inferno', 'seismic': 'seismic', 'hot': 'hot', 'cool': 'cool', 'RdBu': 'RdBu',
            'plasma': 'plasma', 'viridis': 'viridis', 'YlGnBu': 'YlGnBu',
            'winter': 'winter', 'Blues': 'Blues', 'Reds': 'Reds', 'Greys': 'Greys'}

# default random data!
array = np.random.random((100, 100))


xlow = 1.0
xhigh = 1.65
ylow = 0.0
yhigh = 1.0


def plot_limit_cycle_heatmap(save_path = path, save_name = '\heatmap.jpg', data = array, zvar = 'inf', xpar = 'risk', ypar = 'misinformation', xlow = 1, xhigh = 2, ylow = 0, yhigh = 1, cmap = cmap_bin['hot'], show_bool = True):
    """
    function to generte heatmaps with user input data
    :param save_path: path to save the figure
    :param save_name: file name unique to the figure
    :param data: user input numpy data
    :param xpar: parameter for the x in lc boundary
    :param ypar: parameter for the y in lc boundary
    :param xlow: parameter used for low value of x
    :param xhigh: parameter used for high value of x
    :param ylow: parameter used for low value of y
    :param yhigh: parameter used for high value of y
    :param cmap: color used for the heatmap; use one from the dictionary
    :param show_bool: boolean to show the figure
    :return: plotted heatmap!
    """
    xlimit = [xlow, xhigh]
    ylimit = [ylow, yhigh]

    file_vac = f"{xpar}_{ypar}_vac.txt"
    file_bad = f"{xpar}_{ypar}_bad.txt"
    file_inf = f"{xpar}_{ypar}_inf.txt"
    file_xlimit = f"lc_{xpar}_{ypar}_xlimit.txt"
    file_ylimit = f"lc_{xpar}_{ypar}_ylimit.txt"

    data_vac = np.loadtxt(file_vac, delimiter=',', dtype=float)
    data_bad = np.loadtxt(file_bad, delimiter=',', dtype=float)
    data_inf = np.loadtxt(file_inf, delimiter=',', dtype=float)

    if zvar == 'vac':
        data = data_vac
        cbar_lab = 'Vaccinated'
        lc_col = 'r'
    elif zvar == 'inf':
        data = data_inf
        cbar_lab = 'Infected'
        lc_col = 'k'
    elif zvar == 'bad':
        data = data_bad
        cbar_lab = 'Bad'
        lc_col = 'b'
    else:
        data = data

    #xlimit = np.loadtxt(file_xlimit, delimiter=',', dtype=float)
    #ylimit = np.loadtxt(file_ylimit, delimiter=',', dtype=float)

    n = len(data)
    min_data = np.min(data)
    max_data = np.max(data)
    v_min = np.max([min_data, 0])
    v_max = np.min([max_data, 1])
    xlimit = [xlow, xhigh]
    ylimit = [ylow, yhigh]
    par_x_bin = np.linspace(xlimit[0], xlimit[-1], n)
    par_y_bin = np.linspace(ylimit[0], ylimit[-1], n)
    print(par_y_bin)
    row_label = [str(np.round(x0, 3)) for x, x0 in enumerate(par_x_bin)]
    col_label = [str(np.round(x0, 3)) for x, x0 in enumerate(par_y_bin)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if xpar == 'risk':
        xlab = '$r$'
    elif xpar == 'education':
        xlab = r'$\epsilon$'
    elif xpar == 'misinformation':
        xlab = r'$\mu$'
    elif xpar == 'infection_good':
        xlab = r'$\chi$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}$'
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
        ylab = r'$\epsilon$'
    elif ypar == 'misinformation':
        ylab = r'$\mu$'
    elif ypar == 'infection_good':
        ylab = r'$\chi$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}$'
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

    # generate spaces for col_label and row_label
    for z in range(0, len(row_label)):
        if z % 4 != 0:
            print(z % 4)
            row_label[z] = ''
            col_label[z] = ''



    #df = pd.DataFrame(data, columns = col_label, index = row_label)
    #s = sns.heatmap(df, cmap = cmap)
    #h_x = (xlimit[-1] - xlimit[0]) / 8
    #s.set_yticklabels(np.arange(xlimit[0], xlimit[-1], h_x))

    #s.set_xticklabels(col_label)
    #s.set_yticklabels(row_label)

    file_x = f"lc_{xpar}_{ypar}_xsol.txt"
    file_y = f"lc_{xpar}_{ypar}_ysol.txt"
    lc_x = np.loadtxt(file_x, delimiter=',', dtype=float)
    lc_y = np.loadtxt(file_y, delimiter=',', dtype=float)

    # normalize x and y array's!
    lc_x_norm = (lc_x - np.min(lc_x)) / (np.max(lc_x) - np.min(lc_x))
    lc_y_norm = (lc_y - np.min(lc_y)) / (np.max(lc_y) - np.min(lc_y))




    # fill in between to obtain the limit cycle boundary
    # generate a plot for the limit cycle boundary!
    #sns.lineplot(x = lc_y_norm, y = lc_x_norm, color = 'b', lw=5)
    # plt.fill(lc_x, lc_y, 'k')
    # plt.xlabel(xlab, fontsize=18)
    # plt.ylabel(ylab, fontsize=18)
    # plt.xlim([0, xmax])
    # plt.ylim([0, ymax])
    # file_name = f"\lc_{xpar}_{ypar}.jpeg"
    # plt.savefig(path + file_name, dpi=300)
    # plt.show()
    ax.plot(lc_x, lc_y, color = 'b', lw = 2)
    min_data = np.min(data)
    max_data = np.max(data)
    v_min = np.max([min_data, 0])
    v_max = np.min([max_data, 1])
    im = ax.imshow(data, cmap = cmap, vmin = v_min, vmax = v_max,
                   extent = [xlimit[0], xlimit[-1], ylimit[0], ylimit[-1]])

    #plt.xlim(xlimit[0], xlimit[-1])
    #plt.ylim(ylimit[0], ylimit[-1])

    #plt.xlabel(xlab, fontsize = 18)
    #plt.ylabel(ylab, fontsize = 18)




    if show_bool == True:
        file_name = save_path + save_name
        plt.savefig(file_name, dpi = 300)
        plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize = (12,8), subplot_kw={'aspect': 'equal'})

    #ax1.set_yticks([int(j) for j in range(-4, 5)])
    #ax1.set_xticks([int(j) for j in range(-4, 5)])

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(15)
    for tick in ax1.get_xticklines() + ax1.get_yticklines():
        tick.set_markeredgewidth(2)
        tick.set_markersize(6)

    im = ax1.imshow(data, cmap=cmap,
                    extent = [xlimit[0], xlimit[-1], ylimit[0], ylimit[-1]])
    ax1.set_xlim(xlimit[0], xlimit[-1])
    ax1.set_ylim(ylimit[0], ylimit[-1])

    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)

    #ax1.yaxis.set_tick_params(labelsize='xx-large')
    #ax1.xaxis.set_tick_params(labelsize='xx-large')


    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_lab, fontsize = 18)
    ax1.plot(lc_x, lc_y, color = lc_col, lw = 5)
    ax1.fill(lc_x, lc_y, color = lc_col, alpha = 0.25)
    plt.show()

    plt.close(fig)

    return lc_x, lc_y


# In[27]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

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

eq1_risk_h1_ss = { 'x1':  0.16525533433723974,
                    'x2':  0.4608116685077331,
                    'x3':  0.09068387293929851,
                    'x4':  0.14189412776170351,
                    'x5':  0.0003737491664098206}

eq1_risk_h1_par_dict = {'risk': 1.6352957874550589, 'protection': 0.90,
                    'recovery': 0.07, 'belief': 1.0,
                    'education': 0.33, 'misinformation': 0.10,
                    'infection_good': 0.048, 'infection_bad': 0.37}

par_dict_default = eq1_risk_h1_par_dict
ss_dict_default = eq1_risk_h1_ss


def plot_misinformation_bifurcation(par_dict=par_dict_default, ics_dict=ss_dict_default, special_point='H2', tend=5_000,
                               eps0=0.15):
    """
    function to plot codim-1 bifurcation, nullclines, and perturbed time evolution
    :param par_dict: dictionary of parameters used to simulate the bifurcation
    :param ics_dict: dictionary of initial condtiions used to simulate the bifurcation
    :param special_point: point to grab the data on the codim-1 bifurcation
    :param tend: final time used for pointset
    :param eps0: value to perturb the steady states as ics in plotting time
    :return: PC_misinformation the continuation curve that contains all the data
    """
    # generate the ode
    ode = generate_ode(par_dict, ics_dict=ics_dict, tf=500)

    # generate a pointset!
    pts, ss_dict = generate_pointset(ode, save_bool=True)

    # ------------- risk -------------------------
    # generate risk bifurcation!
    PC_misinformation = generate_misinformation_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_misinformation, curve='EQmisinformation', special_point=special_point,
                                           par_dict=par_dict,
                                           par='misinformation')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='misinformation',
                                            random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.6,
                    n_bin=200, par='misinformation')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.50, yhigh=0.5,
                    n_bin=200, par='misinformation')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.75,
                    n_bin=200, par='misinformation')

    return PC_misinformation, par_dict, ss_dict, data


# In[28]:


import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# get dictionary corresponding to multiple parameters and initial conditions!
# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
            'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927



path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# get dictionary corresponding to multiple parameters and initial conditions!
# initial parameter definition!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
            'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


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
                        par_dict_2=eq1_lp1_par_dict, evecs_bool=False, evecs=None, ics_dict={}, par = 'risk', z0 = '', w0 = 0):
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

        x_traj = x1_traj[w0:]
        y_traj = x2_traj[w0:]

        dx_x = 'x1'  # used for quiver
        dx_y = 'x2'

        x_ss = x1_ss  # steady state value on x
        y_ss = x2_ss  # steady state value on y

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.95 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.05 * np.max(x_traj)
        if ylow == None:
            ylow = 0.95 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.05 * np.max(y_traj)

        # generate arrays
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

        x_traj = x2_traj[w0:]
        y_traj = x3_traj[w0:]

        dx_x = 'x2'  # used for quiver
        dx_y = 'x3'

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.85 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.15 * np.max(x_traj)
        if ylow == None:
            ylow = 0.85 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.15 * np.max(y_traj)

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

        x_traj = x3_traj[w0:]
        y_traj = x4_traj[w0:]

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.85 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.15 * np.max(x_traj)
        if ylow == None:
            ylow = 0.85 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.15 * np.max(y_traj)

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

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.95 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.05 * np.max(x_traj)
        if ylow == None:
            ylow = 0.95 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.05 * np.max(y_traj)

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

    elif option == 'E':
        # phi vs v
        a = 5  # eigenvector component in x
        b = 4  # eigenvector component in y

        xlab = r'$\phi$'  # label for x axis
        ylab = r'$V$'  # label for y axis

        xnull_lab = r'$N_{\phi}$'  # label for legend
        ynull_lab = r'$N_{V}$'

        x_traj = x5_traj[w0:]
        y_traj = x4_traj[w0:]

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.95 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.05 * np.max(x_traj)
        if ylow == None:
            ylow = 0.95 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.05 * np.max(y_traj)

        dx_x = 'x5'  # used for quiver
        dx_y = 'x4'

        x_ss = x5_ss  # steady state value on x
        y_ss = x4_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        x_null = (ig_ss + x3_ss + a * ig_ss) / (risk - a * (1 - protection) * infection_good * x3_ss / x_array)

        # y null --> solve for x = x5
        y_null = (1 - protection) * x3_ss * x_array / x1_ss

    # generate a phase field
    if quiv_bool:
        x, y = np.linspace(xlow, xhigh, 15), np.linspace(ylow, yhigh, 15)
        x1, y1 = np.meshgrid(x, y)
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict=par_dict, ss_dict=ss_dict, xvar=dx_x, yvar=dx_y)
        # normalize growth rate!
        dx = dx1 / np.sqrt(dx1**2 + dy1**2);
        dy = dy1 / np.sqrt(dx1**2 + dy1**2);
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1  # avoid division of zero
        dx /= M
        dy /= M  # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot='mid', cmap = 'RdBu')

    if 0 == '':
        z = int(len(x1_traj) / 4)
    else:
        z0 = 0
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
    plt.plot(x_traj[0:], y_traj[0:], 'k')
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.legend()
    file_name = f"\{par}_{dx_x}_{dx_y}_nullcline.jpeg"
    plt.savefig(path + file_name, dpi = 300)
    plt.show()


# In[29]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


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

# dictionaries for parameters obtained on the LP1 previously!
eq1_risk_lp1_par_dict = {'risk': 0.12952930209570576, 'protection': 0.90,
                         'recovery': 0.07, 'belief': 1.0,
                         'education': 0.33, 'misinformation': 0.10,
                         'infection_good': 0.048, 'infection_bad': 0.37}

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


def plot_protection_bifurcation(par_dict=eq1_risk_lp1_par_dict, ics_dict=eq1_risk_lp1_ss, special_point='LP1', tend=5_000, eps0=0.15):
    """
    function to plot codim-1 bifurcation, nullclines, and perturbed time evolution
    :param par_dict: dictionary of parameters used to simulate the bifurcation
    :param ics_dict: dictionary of initial conditions used to simulate the bifurcation
    :param special_point: point to grab the data on the codim-1 bifurcation
    :param tend: final time used for pointset
    :param eps0: value to perturb the steady states as ics in plotting time
    :return: PC_protection the continuation curve that contains all the data
    """
    # generate the ode
    ode = generate_ode(par_dict, ics_dict=ics_dict, tf=500)

    # generate a pointset!
    pts, ss_dict = generate_pointset(ode, save_bool=True)

    # ------------- risk -------------------------
    # generate risk bifurcation!
    PC_protection = generate_protection_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_protection, curve='EQprotection', special_point=special_point, par_dict=par_dict,
                                           par='protection')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='protection', random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.6,
                    n_bin=200, par='protection')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.50, yhigh=0.5,
                    n_bin=200, par='protection')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.75,
                    n_bin=200, par='protection')

    return PC_protection, par_dict, ss_dict, data




# In[30]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


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

# dictionaries for parameters obtained on the LP1 previously!
eq1_risk_lp1_par_dict = {'risk': 0.12952930209570576, 'protection': 0.90,
                         'recovery': 0.07, 'belief': 1.0,
                         'education': 0.33, 'misinformation': 0.10,
                         'infection_good': 0.048, 'infection_bad': 0.37}

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


def plot_recovery_bifurcation(par_dict=eq1_h1_par_dict, ics_dict=eq1_h1_ss, special_point='H2',
                                tend=5_000, eps0=0.15):
    """
    function to plot codim-1 bifurcation, nullclines, and perturbed time evolution
    :param par_dict: dictionary of parameters used to simulate the bifurcation
    :param ics_dict: dictionary of initial conditions used to simulate the bifurcation
    :param special_point: point to grab the data on the codim-1 bifurcation
    :param tend: final time used for pointset
    :param eps0: value to perturb the steady states as ics in plotting time
    :return: PC_recovery the continuation curve that contains all the data
    """
    # generate the ode
    ode = generate_ode(par_dict, ics_dict=ics_dict, tf=500)

    # generate a pointset!
    pts, ss_dict = generate_pointset(ode, save_bool=True)

    # ------------- risk -------------------------
    # generate risk bifurcation!
    PC_recovery = generate_recovery_bifurcation(ode, ics_dict=ics_dict, par_dict=par_dict, tend=300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_recovery, curve='EQrecovery', special_point=special_point,
                                           par_dict=par_dict,
                                           par='recovery')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict=par_dict, ss_dict=ss_dict, tend=tend, par='recovery',
                                            random_bool=True,
                                            eps=eps0)

    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option='A', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.6,
                    n_bin=200, par='recovery')
    # sb vs ib nullcline
    plot_nullclines(option='B', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.50, yhigh=0.5,
                    n_bin=200, par='recovery')
    # ib vs v nullcline
    plot_nullclines(option='C', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh=0.25, yhigh=0.75,
                    n_bin=200, par='recovery')

    return PC_recovery, par_dict, ss_dict, data




# In[31]:


import PyDSTool as dst
from PyDSTool import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

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
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

# dictionaries for parameters obtained on the LP1 previously!
eq1_risk_lp1_par_dict = {'risk': 0.12952930209570576, 'protection': 0.90,
                         'recovery': 0.07, 'belief': 1.0,
                         'education': 0.33, 'misinformation': 0.10,
                         'infection_good': 0.048, 'infection_bad': 0.37}

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


def plot_risk_bifurcation(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, special_point = 'H1', tend = 5_000, eps0 = 0.15):
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


    # ------------- risk -------------------------
    # generate risk bifurcation!
    PC_risk = generate_risk_bifurcation(ode, ics_dict = ics_dict, par_dict = par_dict, tend = 300)

    # get the data
    if special_point != 'BP1':
        par_dict, ss_dict, data = get_data(PC_risk, curve = 'EQrisk', special_point = special_point, par_dict = par_dict, par = 'risk')
    # use EQrisk2 and BP1 (see generate_risk_bifurcation) for more details!
    else:
        par_dict, ss_dict, data = get_data(PC_risk, curve='EQrisk2', special_point=special_point,
                                           par_dict=par_dict, par='risk')

    # generate a pointset and plot the time around the bifurcation!
    pts = plot_time_perturbed_steady_states(PAR_dict = par_dict, ss_dict = ss_dict, tend = tend, par = 'risk', random_bool = True, eps = eps0)


    # plot a few nulllclines
    # sg vs sb nullcline
    plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.25, yhigh = 0.6, n_bin = 200, par = 'risk')
    # sb vs ib nullcline
    plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.50, yhigh = 0.5, n_bin = 200, par = 'risk')
    # ib vs v nullcline
    plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict, ss_dict = ss_dict, evecs_bool = False, xhigh = 0.25, yhigh = 0.75, n_bin = 200, par = 'risk')


    return PC_risk, par_dict, ss_dict, data




# In[32]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams['font.size'] = 18

# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0


# using initial conditions and parameters generated previously!
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37,
                'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
            'ace': 0}

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
            'infection_good': 0.048, 'infection_bad': 0.37,
              'ace': 0}


# this is the steady state value for misinormation starting from the hopf on the risk bifurcation

eq2_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927


ss_dict = ss_bp_r
par_dict = par_hopf_r

def plot_time_perturbed_steady_state(PAR_dict=par_dict, ss_dict=ss_dict, tend=10_000, par='risk',
                                     random_bool=True, eps=0.01):
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
        sgn = np.sign(np.random.rand(1) - 1 / 2)[0]
        ss_dict_perturbed[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1))[0]  # perturb new steady state
    # generate new system
    """ode = generate_ode(PAR_dict, ss_dict_perturbed, tf=tend)  # generate a pointset
    pts = ode.compute('sample').sample(dt=1)
    # plot the variables
    t = pts['t']
    sg = pts['x1']
    sb = pts['x2']
    ib = pts['x3']
    v = pts['x4']
    phi = pts['x5']"""

    z, t = gen_sys(par_dict = PAR_dict, ics_dict = ss_dict_perturbed, tf = tend)
    sg, sb, ib, v, phi = z.T

    pts = {'t': t, 'x1': sg,
           'x2': sb, 'x3': ib,
           'x4': v, 'x5': phi}

    # auxillary
    bad = sb + ib
    good = 1 - (sb + ib + v)
    infected = 1 - (sg + sb + v)
    healthy = sg + sb + v
    ig = 1 - (sb + sg + ib + v)

    # plotting equations of state
    plt.plot(t, sb, 'r', label=r'$S_B$', lw = 2)
    plt.plot(t, ib, 'r--', label=r'$I_B$', lw = 2)
    plt.plot(t, sg, 'b', label=r'$S_G$', lw = 2)
    plt.plot(t, v, 'g', label='$V$', lw = 2)
    plt.plot(t, phi, 'k', label='$\phi$', lw = 2)
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.legend()
    par_lab = par.title()
    file_name = f"\{par_lab}_time_series.jpeg"
    plt.savefig(path + file_name, dpi=300)
    plt.show()

    # plot the effective reproductive number
    reff_num = par_dict['infection_bad'] * sb + (1 - par_dict['protection']) * par_dict['infection_good'] * v
    reff_den = par_dict['recovery'] + par_dict['education'] * good
    reff = reff_num / reff_den
    reff = 1 + (1 - par_dict['protection']) * par_dict['infection_good'] * v + par_dict['infection_bad'] * sb - par_dict['recovery'] -  par_dict['education'] * good
    plt.plot(t, reff, 'b', lw = 2)
    plt.xlabel('t (Days)')
    plt.ylabel('$R_{eff}$')
    par_lab = par.title()
    file_name = f"\{par_lab}_reff.jpeg"
    plt.savefig(path + file_name, dpi=300)
    plt.show()


    return pts


# In[43]:


# generate a new time series with low spread of misinformation
par_dict_new = eq1_h1_par_dict
par_dict_new['misinformation'] = 0.000001
par_dict_new['ace'] = 0




# generate the time series and z values
z, t = gen_sys(par_dict = par_dict_new, ics_dict = eq1_h1_ss, tf = 100)

# unpack key variables
sg = z[:,0]
sb = z[:,1]
ib = z[:,2]
v = z[:,3]
phi = z[:,4]

# generate a pointset used for the plotting the nullcines
pts = {'x1': sg, 'x2': sb, 'x3': ib, 'x4': v, 'x5': phi, 't': t}

# plot the time series
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict_new, ss_dict = eq1_h1_ss, tend = 20_000, par = 'misinformation',
                                        random_bool = False, eps = 0.0001)


print(pts)


ss_new = {'x1': sg[-1], 
         'x2': sb[-1],
         'x3': ib[-1], 
         'x4': v[-1], 
         'x5': phi[-1]}


# In[33]:


plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_new, ss_dict = ss_new,quiv_bool = True, evecs_bool = False,  xhigh = 1, yhigh = 1, xlow = 0, ylow = 0, n_bin = 200, par = 'misinformation', z0 = 0)
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_new, ss_dict = ss_new, quiv_bool = True, evecs_bool = False,  xhigh = 1, yhigh = 1, xlow = 0, ylow = 0, n_bin = 200, par = 'misinformation', z0 = 0)
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_new, ss_dict = ss_new, quiv_bool = True,  xhigh = 1e-15, yhigh = 0.25, xlow = -1e-15, ylow = 0, n_bin = 200, par = 'misinformation', z0 = 0)
# ib vs v nullcline
#plot_nullclines(option='E', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh = 1, yhigh = 1, xlow = 0, ylow = 0, n_bin=200, par='education')


# In[52]:


# generate a new time series with low spread of misinformation
par_dict_new2 = eq1_h1_par_dict
par_dict_new2['misinformation'] = 0.55
par_dict_new2['ace'] = 0

print(par_dict_new2)

par_dict_new2 = {'recovery': 0.07, 'belief': 1.0, 'risk': 1.635295791362042,
                 'protection': 0.9, 'education': 0.33, 'misinformation': 0.25,
                 'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}




# generate the time series and z values
z, t = gen_sys(par_dict = par_dict_new2, ics_dict = eq1_h1_ss, tf = 20_000)

# unpack key variables
sg = z[:,0]
sb = z[:,1]
ib = z[:,2]
v = z[:,3]
phi = z[:,4]

plt.plot(z)
plt.show()

# generate a pointset used for the plotting the nullcines
pts = {'x1': sg, 'x2': sb, 'x3': ib, 'x4': v, 'x5': phi, 't': t}

# plot the time series
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict_new2, ss_dict = eq1_h1_ss, tend = 20_000, par = 'misinformation',
                                        random_bool = True, eps = 0.1)


ss_new = {'x1': sg[-1], 
         'x2': sb[-1],
         'x3': ib[-1], 
         'x4': v[-1], 
         'x5': phi[-1]}


# In[46]:


plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation')
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True,  xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True,  xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation')
# ib vs v nullcline
#plot_nullclines(option='E', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh = 1, yhigh = 1, xlow = 0, ylow = 0, n_bin=200, par='education')


# In[50]:


# generate a new time series with low spread of misinformation
par_dict_new2 = eq1_h2_par_dict
#par_dict_new2['misinformation'] = 0.55
par_dict_new2['ace'] = 0
eq1_h2_par_dict['risk'] = 0.342001918986597
eq1_h2_par_dict['misinformation'] = 0.066

# --
eq1_h2_ss = {'x1':  0.20993799319537826,
             'x2':  0.48221092580065467,
             'x3':  0.07073195800020968,
             'x4':  0.1168184163664055,
             'x5':  0.0001889196754370767}

# parameters for branching point at risk bifurcation!
par_bp_r = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.34021985, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}
eq1_h1_par_dict = par_bp_r
eq1_h1_par_dict['risk'] = 1.635295791362042
eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927



print(par_dict_new2)

par_dict_new2 = eq2_h2_par

"""par_dict_new2 = {'recovery': 0.07, 'belief': 1.0, 'risk': 1.635295791362042,
                 'protection': 0.9, 'education': 0.33, 'misinformation': 0.25,
                 'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}"""




# generate the time series and z values
z, t = gen_sys(par_dict = par_dict_new2, ics_dict = eq1_h2_ss, tf = 20_000)

# unpack key variables
sg = z[:,0]
sb = z[:,1]
ib = z[:,2]
v = z[:,3]
phi = z[:,4]


# generate a pointset used for the plotting the nullcines
pts = {'x1': sg, 'x2': sb, 'x3': ib, 'x4': v, 'x5': phi, 't': t}

# plot the time series
pts = plot_time_perturbed_steady_state(PAR_dict = par_dict_new2, ss_dict = eq1_h2_ss, tend = 20_000, par = 'misinformation',
                                        random_bool = True, eps = 0.1)

sg = pts['x1']
sb = pts['x2']
ib = pts['x3']
v = pts['x4']
phi = pts['x5']


from scipy.signal import argrelextrema

sg_high = sg[argrelextrema(sg[10_000:], np.greater)[0]]
sg_low = sg[argrelextrema(sg[10_000:], np.less)[0]]
sb_high = sb[argrelextrema(sb[10_000:], np.greater)[0]]
sb_low = sb[argrelextrema(sb[10_000:], np.less)[0]]
ib_high = ib[argrelextrema(ib[10_000:], np.greater)[0]]
ib_low = ib[argrelextrema(ib[10_000:], np.less)[0]]
v_high = v[argrelextrema(v[10_000:], np.greater)[0]]
v_low = v[argrelextrema(v[10_000:], np.less)[0]]
phi_high = phi[argrelextrema(phi[10_000:], np.greater)[0]]
phi_low = phi[argrelextrema(phi[10_000:], np.less)[0]]

print(np.mean(sg_high))

ss_new = {'x1': (np.average(sg_high) +  np.average(sg_low)) / 2, 
         'x2': (np.mean(sb_high) + np.mean(sb_low)) / 2,
         'x3': (np.mean(ib_high) + np.mean(ib_low)) / 2, 
         'x4': (np.mean(v_high) + np.mean(v_low)) / 2, 
         'x5': (np.mean(phi_high) + np.mean(phi_low)) / 2}

print(ss_new)


# In[51]:


plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation', z0 = 0)
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True,  xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_new2, ss_dict = ss_new, quiv_bool = True,  xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'misinformation')
# ib vs v nullcline


# In[46]:


# generate a risk bifurcation and determine the steady state values
par_dict['ace'] = 0
ode = generate_ode(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, tf = 2_000)
PC_education = generate_education_bifurcation(ode, ics_dict = eq1_h2_ss, par_dict = eq1_h2_par_dict, max_points = 1000, tend = 500) 


# In[34]:


par_dict_h1, ss_dict_h1, data_h1 = get_data(PC_education, curve='EQeducation', special_point='H1', par='education', par_dict=eq1_h1_par_dict)


# In[426]:


par_dict_h2, ss_dict_h2, data_h2 = get_data(PC_education, curve='EQeducation', special_point='H2', par='education', par_dict=eq1_h1_par_dict)


# In[427]:


par_dict_h3, ss_dict_h3, data_h3 = get_data(PC_education, curve='EQeducation', special_point='H3', par='education', par_dict=eq1_h1_par_dict)


# In[428]:


par_dict_h4, ss_dict_h4, data_h4 = get_data(PC_education, curve='EQeducation', special_point='H4', par='education', par_dict=eq1_h1_par_dict)


# In[433]:


par_dict_lp1, ss_dict_lp1, data_lp1 = get_data(PC_education, curve='EQeducation', special_point='LP1', par='education', par_dict=eq1_h1_par_dict)

par_dict_h2


# In[415]:


# obtain the temporal evolution of about the sub-critical andronov hopf bifurcation!
# easy to obtain, all we gotta do, is start from the hopf bifurcation, then plot into the time series 
# --- easy ---
pts = plot_time_perturbed_steady_state(PAR_dict=par_dict_lp1, ss_dict=ss_dict_lp1, tend=10_000, par='education',
                                     random_bool=True, eps=0.1)


sg = pts['x1']
sb = pts['x2']
ib = pts['x3']
v = pts['x4']
phi = pts['x5']


from scipy.signal import argrelextrema

sg_high = sg[argrelextrema(sg[2_000:], np.greater)[0]]
sg_low = sg[argrelextrema(sg[2_000:], np.less)[0]]
sb_high = sb[argrelextrema(sb[2_000:], np.greater)[0]]
sb_low = sb[argrelextrema(sb[2_000:], np.less)[0]]
ib_high = ib[argrelextrema(ib[2_000:], np.greater)[0]]
ib_low = ib[argrelextrema(ib[2_000:], np.less)[0]]
v_high = v[argrelextrema(v[2_000:], np.greater)[0]]
v_low = v[argrelextrema(v[2_000:], np.less)[0]]
phi_high = phi[argrelextrema(phi[2_000:], np.greater)[0]]
phi_low = phi[argrelextrema(phi[2_000:], np.less)[0]]

print(np.mean(sg_high))

ss_new = {'x1': (np.average(sg_high) +  np.average(sg_low)) / 2, 
         'x2': (np.mean(sb_high) + np.mean(sb_low)) / 2,
         'x3': (np.mean(ib_high) + np.mean(ib_low)) / 2, 
         'x4': (np.mean(v_high) + np.mean(v_low)) / 2, 
         'x5': (np.mean(phi_high) + np.mean(phi_low)) / 2}


# In[416]:


plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_h1, ss_dict = ss_dict_h1, quiv_bool = True, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'education', w0 = 0, z0 = 0)
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_h1, ss_dict = ss_dict_h1, quiv_bool = True, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 10_000, par = 'education', w0 = 0, z0 = 0)
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_h1, ss_dict = ss_dict_h1, quiv_bool = True, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin = 200, par = 'education', w0 = 0, z0 = 0)
# ib vs v nullcline
plot_nullclines(option='E', PTS=pts, par_dict=par_dict, ss_dict=ss_dict, evecs_bool=False, xhigh = None, yhigh = None, xlow = None, ylow = None, n_bin=200, par='education', z0 = 0, w0 = 0)


# In[162]:


par_dict_h1


# In[264]:


eq1_h1_par_dict['misinformation'] = 0.10

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict['risk'] = 1.635


# In[265]:


# generate a risk bifurcation and determine the steady state values
eq1_h1_par_dict['ace'] = 0
eq1_h1_par_dict['education'] = 0.33
ode = generate_ode(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, tf = 20_000)
PC_risk = generate_risk_bifurcation(ode, ics_dict = eq1_h1_ss, par_dict = eq1_h1_par_dict, max_points = 500, tend = 1000)


# In[256]:


# get the data
par_dict_lp1, ss_dict_lp1, data_lp1 = get_data(PC_risk, curve='EQrisk', special_point='LP1', par='risk', par_ext='', par_dict=eq1_h1_par_dict)
par_dict_bp1, ss_dict_bp1, data_bp1 = get_data(PC_risk, curve='EQrisk2', special_point='BP1', par='risk', par_ext='', par_dict=eq1_h1_par_dict)


# In[262]:


# plot the lp1
pts = plot_time_perturbed_steady_state(PAR_dict=par_dict_lp1, ss_dict=ss_dict_lp1, tend=10_000, par='risk',
                                     random_bool=True, eps=0.1e-4)

print(par_dict_lp1)


# In[261]:


plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_lp1, ss_dict = ss_dict_lp1, evecs_bool = False, xhigh = None, yhigh = None, xlow = 0, ylow = None, n_bin = 200, par = 'risk', z0 = 0)
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_lp1, ss_dict = ss_dict_lp1, evecs_bool = False, xhigh = None, yhigh = None, xlow = 0, ylow = None, n_bin = 200, par = 'risk', z0 = 0)
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_lp1, ss_dict = ss_dict_lp1, evecs_bool = False, xhigh = None, yhigh = None, xlow = 0, ylow = None, n_bin = 200, par = 'risk', z0 = 0)


# In[255]:


# plot the lp1
pts = plot_time_perturbed_steady_state(PAR_dict=par_dict_bp1, ss_dict=ss_dict_bp1, tend=20_000, par='risk',
                                     random_bool=True, eps=0.001)

print(par_dict_lp1)
print(ss_dict_lp1)


# In[186]:



plot_nullclines(option = 'A', PTS = pts, par_dict = par_dict_bp1, ss_dict = ss_dict_bp1, evecs_bool = False, xhigh = 0.4, yhigh = 0.8, xlow = 0, ylow = 0, n_bin = 200, par = 'risk')
# sb vs ib nullcline
plot_nullclines(option = 'B', PTS = pts, par_dict = par_dict_bp1, ss_dict = ss_dict_bp1, evecs_bool = False, xhigh = 1, yhigh = 1, xlow = 0, ylow = 0, n_bin = 200, par = 'risk')
# ib vs v nullcline
plot_nullclines(option = 'C', PTS = pts, par_dict = par_dict_bp1, ss_dict = ss_dict_bp1, evecs_bool = False, xhigh = 0.3, yhigh = 0.7, xlow = 0, ylow = 0, n_bin = 200, par = 'risk')


# In[ ]:




