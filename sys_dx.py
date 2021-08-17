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
