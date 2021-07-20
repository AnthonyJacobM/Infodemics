import numpy as np
import matplotlib.pyplot as plt
import generate_ode
import matplotlib as mpl

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


ss_dict = ss_bp_r
par_dict = par_hopf_r

def plot_time_perturbed_steady_state(PAR_dict=par_dict, ss_dict=ss_dict, tend=10_000, par='risk',
                                     random_bool=True, eps=0.01, par_dict=eq1_h1_par_dict):
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
        sgn = np.sign(eps - 1 / 2)
        ss_dict_perturbed[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1))  # perturb new steady state
    # generate new system
    ode = generate_ode(PAR_dict, ss_dict_perturbed, tf=tend)  # generate a pointset
    pts = ode.compute('sample').sample(dt=1)

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
    plt.plot(t, sb, 'r', label=r'$S_B$')
    plt.plot(t, ib, 'r--', label=r'$I_B$')
    plt.plot(t, sg, 'b', label=r'$S_G$')
    plt.plot(t, v, 'b--', label='$V$')
    plt.plot(t, phi, 'm', label='$\phi$')
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

    plt.plot(t, bad, 'r', label=r'$B$')
    plt.plot(t, good, 'k', label=r'$G$')
    plt.plot(t, v, 'b', label=r'$V$')
    plt.xlabel('t (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

    return pts