import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl

# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200

# fontsize is 18
mpl.rcParams[f'font.size'] = 18

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

example()