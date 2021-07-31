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
                      'protection': [0, 1], 'risk': [0, 6],
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