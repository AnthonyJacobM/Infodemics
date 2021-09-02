# a collection of auxillary functions
# import relevant libraries
import PyDSTool as dst
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plot
import matplotlib as mpl
from scipy.integrate import odeint
import pandas as pd
import openpyxl
import xlsxwriter

# change default matplotlib files
# dpi changes resolution of figures
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 200
# fontsize is 18
mpl.rcParams['font.size'] = 18
# linewidth is 2
mpl.rcParams['lines.linewidth'] = 2.0

# initialize parameter dictionary for the system
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
par_bp_r = {'recovery': 0.07, 'belief': 1.0, 'ace': 0,
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



eq1_h2_par_dict = par_bp_r
eq1_h2_par_dict['risk'] = 0.342001918986597

eq1_lp1_par_dict = par_bp_r
eq1_lp1_par_dict['risk'] = 0.1295293020919909

eq1_h1_par_dict = par_bp_r
eq1_h1_par_dict['risk'] = 1.635295791362042


eq1_h2_ss = {'x1': 0.0005386052264346348,
             'x2': 0.18947415697215972,
             'x3': 0.1978608081601073,
             'x4': 0.6035663782158089,
             'x5':  1.064278997774266}

eq2_h2_par = eq1_h1_par_dict
eq2_h2_par['misinformation'] = 0.0660611192767927

par_hopf_r = {'recovery': 0.07, 'belief': 1.0, 'ace': 0,
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



def generate_ode(par_d = par_dict_def, ics_d = ics_dict_def, tend = 1000, rev_bool = False):
    """
    function to generate an ode for the infodemics model
    :param p_d: dictionary of parameters
    :param i_d: dictionary of initial conditions
    :param tend: final time of numerical simisinformationlation
    :param rev_bool: boolean for the revised version of the model (short term == True)
    :return: ode
    """

    # generate DSargs!
    DSargs = dst.args(name='infodemics_rev')

    # unpack parameters!!!!
    recovery = Par(par_d['recovery'], 'recovery')
    belief = Par(par_d['belief'], 'belief')
    risk = Par(par_d['risk'], 'risk')
    protection = Par(par_d['protection'], 'protection')
    education = Par(par_d['education'], 'education')
    misinformation = Par(par_d['misinformation'], 'misinformation')
    infection_good = Par(par_d['infection_good'], 'infection_good')
    infection_bad = Par(par_d['infection_bad'], 'infection_bad')
    ace = Par(par_d['ace'], 'ace')

    DSargs.pars = [recovery, belief, risk, protection, education,
                   misinformation, infection_good, infection_bad, ace]

    # generate state variables!
    x1 = Var('x1')  # sg
    x2 = Var('x2')  # sb
    x3 = Var('x3')  # ib
    x4 = Var('x4')  # v
    x5 = Var('x5')  # phi

    # generate initial condiitons!
    DSargs.ics = ics_d

    # generate time domain!
    DSargs.tdomain = [0, tend]

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
    # long run model:
    if rev_bool == False:
        # generate right hand side of the differential equations!
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (
                    x5 + (infection_good + misinformation) * x3 + misinformation * x2)
        x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (
                    1 - protection) * infection_good * x4)
        x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
        x5rhs = belief * x5 * (1 - x5) * (ace * (1 - x2 - x3 - x4) + (1 - x1 - x2 - x4) - risk * x4)

    # short run model
    else:
        # generate right hand side of the differential equations!
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) + (1 - protection) * x4 - x1 * (x5 + (infection_good + misinformation) * x3 + misinformation * x2)
        x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)))
        x4rhs = x5 * x1 - (1 - protection) * x4
        x5rhs = belief * x5 * (1 - x5) * (ace * (1 - x2 - x3 - x4) + (1 - x1 - x2 - x4) - risk * x4)

    DSargs.varspecs = {'x1': x1rhs, 'x2': x2rhs,
                       'x3': x3rhs, 'x4': x4rhs,
                       'x5': x5rhs}

    DSargs.fnspecs = {'infected': (['t', 'sg', 'sb', 'v'], '1 - sg - sb - v'),
                      'bad': (['t', 'sb', 'ib'], 'sb + ib'),
                      'good': (['t', 'sb', 'ib', 'v'], '1 - sb - ib - v')}

    # change any default numerical parameters
    DSargs.algparams = {'max_pts': 10_000, 'stiff': True}

    # generate the ordinary differential equations!
    ode = dst.Vode_ODEsystem(DSargs)

    return ode

# revised
tend = 10_000 # final time of the integration
def gen_sys(par_dict = par_dict_def, ics_dict = ics_dict_def, tf = tend, rev_bool = False):
    """
    function to generate a system used for integration
    :param par_dict: dictionary for parameters
    :param ics_dict: dicitonary for initial conditions
    :param tf: final time
    :return: the solution
    """

    # unpack the dictionaries
    recovery = par_dict['recovery']
    infection_good = par_dict['infection_good']
    infection_bad = par_dict['infection_bad']
    misinformation = par_dict['misinformation']
    education = par_dict['education']
    risk = par_dict['risk']
    protection = par_dict['protection']

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

        if rev_bool == False:
            Y[0] = recovery * (1 - X[0] - X[1] - X[2] - X[3]) - X[0] * (X[4] + (infection_good + misinformation) * X[2] + misinformation * X[1])
            Y[1] = misinformation * X[0] * (X[1] + X[2]) - X[2] * (infection_bad * X[1] - recovery)
            Y[2] = X[2] * (infection_bad * X[1] - recovery - education * ((1 - X[1] - X[2] - X[3])) + (1 - protection) * infection_good * X[3])
            Y[3] = X[4] * X[0] - (1 - protection) * infection_good * X[3] * X[2]
            Y[4] = 1 * X[4] * (1 - X[4]) * (0 * (1 - X[1] - X[2] - X[3]) + X[2] + (1 - X[0] - X[1] - X[2] - X[3]) - risk * X[3])
        else:
            Y[0] = recovery * (1 - X[0] - X[1] - X[2] - X[3]) + (1 - protection) * X[3] - X[0] * (infection_good * X[2] + misinformation * (X[1] + X[2]) + X[4])  # sg
            # gamma * ib + misinformation * sg * (sb + ib) - chi_hat * sb * ib
            Y[1] = recovery * X[2] + misinformation * X[0] * (X[1] + X[2]) - infection_bad * X[1] * X[2]  # sb
            # ib * (chi_hat * sb - gamma - education * (ig + sg))
            Y[2] = X[2] * (infection_bad * X[1] - recovery - education * (1 - X[1] - X[2] - X[3]))  # ib
            # phi * sg - (1 - delta) * v
            Y[3] = X[4] * X[0] - (1 - protection) * X[3]
            # phi * (1 - phi) * (ig + ib -  r * v)
            Y[4] = X[4] * (1 - X[4]) * (1 - X[0] - X[1] - X[3] - risk * X[3])
        return Y

    # integrate the solution
    z, infodict = odeint(sys, x0, t, full_output = True)

    return z, t

# define a function to export the data to excel
def export_data_to_excel(data = None, path = r'C:\Users\antho\Documents\Projects\Infodemics\Code\data', file_name = 'excel_test', sheets = ['StateVariables', 'Parameters']):
    """
    function to export the data to an excel file
    :param data: contains the data for the state variables and the parameter values as two dictionaries for default
    :param path: the path that will be used for the export of the data
    :param file_name: name of the file that will be saved in the excel export
    :return: .xslx file in the data
    """
    # generate new data from an ODE provided the data is NaN type
    if data == None:
        z, t = gen_sys(ics_dict=ics_dict_def, par_dict=par_dict_def)
        sg, sb, ib, v, phi = z.T
        ig = 1 - (sg + sb + v + ib)
        I = ig + ib
        G = sg + ig
        B = sb + ib
        data_1 = {'t': t, 'sg': sg, 'sb': sb, 'ig': ig, 'ib': ib, 'v': v, 'phi': phi, 'G': G, 'B': B, 'I': I}
        data_2 = par_dict_def
        data = [data_1, data_2] # combined data set........
    # generate a data frame given the data
    df1 = pd.DataFrame(data = data[0])
    df2 = pd.DataFrame(data = data[1], index = [0])
    df_list = [df1, df2]


    # export the data to an excel file
    file = f"/{file_name}"
    file_export = path + file + '.xlsx'

    # generate an excel writer
    excel_writer = pd.ExcelWriter(file_export, engine = 'xlsxwriter')

    # loop through the data list and export to an xlsx format:
    for i, df in enumerate(df_list):
        df.to_excel(excel_writer, index = False, sheet_name=sheets[i])

    # save the file
    excel_writer.save()



def sys_dx(X, t = 0, par_dict = par_dict_def, ss_dict = ics_dict_def, xvar = 'x1', yvar  = 'x3', rev_bool = True):
    """
    function to generate the phase field of the state varialbes and parameters
    :param X: 5 dimensional array
    :param t: time
    :param par_dict: dicitonary of parameters
    :param ss_dict: dictioanry of steady state variables
    :param xvar: x variable for the phase space
    :param yvar: y variable for the phase space
    :param rev_bool: Boolean for the short term revised model
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


    # long term model
    if rev_bool == False:
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) - x1 * (x5 + (infection_good + misinformation) * x3 + misinformation * x2)
        x2rhs = misinformation * x1 * (x2 + x3) - x3 * (infection_bad * x2 - recovery)
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (x1 + (1 - x1 - x2 - x3 - x4)) + (1 - protection) * infection_good * x4)
        x4rhs = x5 * x1 - (1 - protection) * infection_good * x4 * x3
        x5rhs = x5 * (1 - x5) * ((1 - x1 - x2 - x4) - risk * x4)



    # short term model
    else:
        x1rhs = recovery * (1 - x1 - x2 - x3 - x4) + (1 - protection) * x4 - x1 * (infection_good * x3 + misinformation * (x2 + x3) + x5)  # sg
        # gamma * ib + misinformation * sg * (sb + ib) - chi_hat * sb * ib
        x2rhs = recovery * x3 + misinformation * x1 * (x2 + x3) - infection_bad * x2 * x3  # sb
        # ib * (chi_hat * sb - gamma - education * (ig + sg))
        x3rhs = x3 * (infection_bad * x2 - recovery - education * (1 - x2 - x3 - x4))  # ib
        # phi * sg - (1 - delta) * v
        x4rhs = x5 * x1 - (1 - protection) * x4
        # phi * (1 - phi) * (ig + ib -  r * v)
        x5rhs = x5 * (1 - x5) * ((1 - x1 - x2 - x4) - risk * x4)

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


def plot_nullclines(option='A', PTS='', par_dict=eq1_h1_par_dict, ss_dict=eq1_h1_ss, n_bin=500, xlow=None, xhigh=None,
                        ylow=None, yhigh=None, quiv_bool=True, w0 = 0, distance = 0.15, z0 = 0, save_bool = False,
                        evecs_bool=False, evecs=None, ics_dict={}, par = 'risk', seq = '', title_bool = False, title = ''):
    """
    function to generate nullclines with a quiver plot field determined using sys_dx function
    :param option: 'A' for SG vs SB, 'B' for SB vs IB, 'C' for IB vs V
    :param PTS: pointset, if '', then generate it
    :param par_dict: dictionary of paramaeters
    :param ss_dict: dicitonary of steay state values
    :param n_bin: number of elements in the np array
    :param xlow: lower bound on x variables
    :param xhigh: upper bound on x variable
    :param ylow: lower bound on y variable
    :param yhigh: upper bound on y variable
    :param quiv_bool: boolean for the quiver plot of the nullcline
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
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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

        x_traj = x3_traj[w0:]
        y_traj = x4_traj[w0:]

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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
    # generate a phase field
    if quiv_bool:
        x, y = np.linspace(xlow, xhigh, 15), np.linspace(ylow, yhigh, 15)
        x1, y1 = np.meshgrid(x, y)
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict=par_dict, ss_dict=ss_dict, xvar=dx_x, yvar=dx_y)
        # normalize growth rate!
        dx = dx1 / np.sqrt(dx1 ** 2 + dy1 ** 2);
        dy = dy1 / np.sqrt(dx1 ** 2 + dy1 ** 2);
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1  # avoid division of zero
        dx /= M
        dy /= M  # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot='mid', cmap='RdBu')

    if z0 == '':
        z = int(len(x1_traj) / 4)
    else:
        z = 0
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

        # plot the eigenvector's starting from the fixed point
        plt.arrow(x_ss, y_ss, v1[0], v1[1], color='b', ls='--')
        plt.arrow(x_ss, y_ss, v2[0], v2[1], color='r', ls='--')

    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    plt.plot(x_traj[0:], y_traj[0:], 'k')
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.legend()
    if seq == '':
        file_name = f"\{par}_{dx_x}_{dx_y}_nullcline.jpeg"
    else:
        file_name = f"\{par}_{dx_x}_{dx_y}_nullcline_{seq}.jpeg"
    if save_bool == True:
        plt.savefig(path + file_name, dpi=100)
    if title_bool == True:
        plt.title(title, fontsize = 18)
    plt.show()



def plot_nullclines_rev(option='A', PTS='', par_dict=eq1_h1_par_dict, ss_dict=eq1_h1_ss, n_bin=500, xlow=None, xhigh=None,
                        ylow=None, yhigh=None, quiv_bool=True, w0 = 0, distance = 0.15, z0 = 0, save_bool = False,
                        evecs_bool=False, evecs=None, ics_dict={}, par = 'risk', seq = '', title_bool = False, title = ''):
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
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

        # generate arrays
        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x2
        x_null = (recovery * ig_ss + (1 - protection) * x4_ss - x_array * (x5_ss + (infection_good + misinformation) * x3_ss)) / (
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
        dx_y = 'x3' # used for quiver

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

        x_ss = x2_ss  # steady state value on x
        y_ss = x3_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x3
        x_null = (misinformation * x1_ss * x_array) / (infection_bad * x_array - recovery - misinformation * x1_ss)
        # y null --> solve for x = x2
        y_null = (recovery + education * (x1_ss + (infection_good + education) * x1_ss * y_array / (recovery - education * y_array)))


    elif option == 'p':
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
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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

    elif option == 'C':
        # ig vs ib
        a = None  # eigenvector component in x
        b = 3  # eigenvector component in y

        xlab = r'$S_G$'  # label for x axis
        ylab = r'$I_B$'  # label for y axis

        xnull_lab = r'$N_{S_G}$'  # label for legend
        ynull_lab = r'$N_{I_B}$'

        x_traj = x1_traj
        y_traj = x3_traj

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = 0.95 * np.min(x_traj)
        if xhigh == None:
            xhigh = 1.05 * np.max(x_traj)
        if ylow == None:
            ylow = 0.95 * np.min(y_traj)
        if yhigh == None:
            yhigh = 1.05 * np.max(y_traj)

        dx_x = 'x1'  # used for quiver
        dx_y = 'x3'

        x_ss = x3_ss  # steady state value on x
        y_ss = x4_ss  # steady state value on y

        x_array = np.linspace(xlow, xhigh, n_bin)
        y_array = np.linspace(ylow, yhigh, n_bin)

        # x null --> solve for y = x4
        x_null = (recovery * ig_ss + (1 - protection) * x5_ss - x_array * (x5_ss + misinformation * x3_ss)) / (misinformation + infection_good)

        # y null --> solve for x = x3
        z = recovery + education * (ig_ss + x1_ss)
        z_null = (misinformation * x_array * (recovery + z)) / (infection_bad * z - infection_bad * (recovery + misinformation * x1_ss)) # with respect to solved for ib and varying sg
        y_null = (infection_bad * y_array * (recovery + misinformation * x1_ss) / (infection_bad  * x3_ss - misinformation * x1_ss)) / (recovery + education * (x1_ss + (infection_good + education) * x1_ss * y_array) / (recovery - education * x3_ss)) - 1

    elif option == 'E':
        # phi vs v
        a = 5  # eigenvector component in x
        b = 4  # eigenvector component in y

        xlab = r'$\phi$'  # label for x axis
        ylab = r'$V$'  # label for y axis

        xnull_lab = r'$N_{\phi}$'  # label for legend
        ynull_lab = r'$N_{V}$'

        x_traj = x3_traj[w0:]
        y_traj = x4_traj[w0:]

        # generate limits on the boundaries of the plot
        if xlow == None:
            xlow = (1 - distance) * np.min(x_traj)
        if xhigh == None:
            xhigh = (1 + distance) * np.max(x_traj)
        if ylow == None:
            ylow = (1 - distance) * np.min(y_traj)
        if yhigh == None:
            yhigh = (1 + distance) * np.max(y_traj)

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
    # generate a phase field
    if quiv_bool:
        x, y = np.linspace(xlow, xhigh, 15), np.linspace(ylow, yhigh, 15)
        x1, y1 = np.meshgrid(x, y)
        dx1, dy1 = sys_dx([x1, y1, x1, y1, x1, y1], par_dict=par_dict, ss_dict=ss_dict, xvar=dx_x, yvar=dx_y)
        # normalize growth rate!
        dx = dx1 / np.sqrt(dx1 ** 2 + dy1 ** 2)
        dy = dy1 / np.sqrt(dx1 ** 2 + dy1 ** 2)
        M = (np.hypot(dx1, dy1))
        M[M == 0] = 1  # avoid division of zero
        dx /= M
        dy /= M  # normalize arrows

        plt.quiver(x, y, dx1, dy1, M, pivot='mid', cmap='RdBu')

    if z0 == '':
        z = int(len(x1_traj) / 4)
    else:
        z = 0
    if option == 'C':
        plt.plot(x_array, x_null, 'b', label=xnull_lab)
        plt.plot(x_array, z_null, 'r', label=ynull_lab)
    else:
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

    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    plt.plot(x_traj[0:], y_traj[0:], 'k')
    plt.xlim([xlow, xhigh])
    plt.ylim([ylow, yhigh])
    plt.legend()
    if seq == '':
        file_name = f"\_revised_{par}_{dx_x}_{dx_y}_nullcline.jpeg"
    else:
        file_name = f"\_revised_{par}_{dx_x}_{dx_y}_nullcline_{seq}.jpeg"
    if save_bool == True:
        plt.savefig(path + file_name, dpi=100)
    if title_bool == True:
        plt.title(title, fontsize = 18)

    plt.show()

def plot_time(z, t, par_dict = par_dict_def, two_d_bool = False):
    """
    function to plot a time series
    :param z: state variables
    :param t: time sequence
    :param par_dict: dictionary for parameters
    :param two_d_bool: boolean for plot of 2 dimension
    :return: ss_d -- dictionary of final time plot
    """
    # unpack the parameter dictionary
    risk = par_dict['risk']
    protection = par_dict['protection']
    infection_bad = par_dict['infection_bad']
    infection_good = par_dict['infection_good']
    misinformation = par_dict['misinformation']
    education = par_dict['education']
    recovery = par_dict['recovery']

    x1, x2, x3, x4, x5 = z.T
    ig = 1 - (x1 + x2 + x3 + x4)

    infected = x3 + ig
    good = x1 + ig
    bad = x2 + x3

    x1_ss = x1[-1]
    x2_ss = x2[-2]
    x3_ss = x3[-1]
    x4_ss = x4[-1]
    x5_ss = x5[-1]

    ss_d = {'x1': x1_ss, 'x2': x2_ss, 'x3': x3_ss, 'x4': x4_ss, 'x5': x5_ss}

    # plot the time evolution of the state variables
    plt.plot(t, x1, 'b', label='SG')
    plt.plot(t, ig, 'b--', label='IG')
    plt.plot(t, x2, 'r', label='SB')
    plt.plot(t, x3, 'r--', label='IB')
    plt.plot(t, x4, 'g', label='V')
    plt.plot(t, x5, 'k', label='$\phi$')
    plt.xlabel('t (Days)', fontsize=18)
    plt.legend()
    plt.show()

    # plot the effective reproductive number
    #reff_num = (infection_bad * x3 * (recovery + misinformation * x1) / (infection_bad * x3 - misinformation * x1))
    #reff_den = (recovery + education * (x1 + (infection_good + education) * x1 * x3) / (recovery - education * x3))
    #reff = reff_num / reff_den
    reff = infection_bad * x2 / (recovery + education * (x1 + ig))
    plt.plot(t, reff, 'b', lw = 2)
    plt.xlabel('t (Days)', fontsize = 18)
    plt.ylabel(r'$R_{e}$', fontsize = 18)
    plt.show()

    # plot the coexpression
    if two_d_bool == True:
        plt.plot(good, infected, 'b')
        plt.xlabel('good')
        plt.ylabel('infected')
        plt.show()

        plt.plot(bad, infected, 'r')
        plt.xlabel('bad')
        plt.ylabel('infected')
        plt.show()

        plt.plot(infected, x4, 'k')
        plt.xlabel('infected')
        plt.ylabel('vaccinated')
        plt.show()

        plt.plot(x3, x4, 'g')
        plt.xlabel('ib')
        plt.ylabel('v')
        plt.show()

    return ss_d


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


def get_data(PC, curve='EQrisk', special_point='H1', par='risk', par_ext='', par_dict=eq1_h1_par_dict):
    """
    function to get_data from the bifurcation plot!
    :param curve: PC[curve]
    :param special_point: PC[curve.getSpecialPoint(special_point)
    :param par_ext: if '' do not use, otherwise there will be an additional parameter
    :param par_dict, dictionary for parameters used in the bifurcation as the baseline
    :return: par_dict, ss_dict, data <--> dictionary of parameters and steady state values, data of the curve
    """
    data = PC[curve].getSpecialPoint(special_point)
    par_dict[par] = data[par]

    if par_ext != '':
        par_dict[par_ext] = data[par_ext]

    ss_dict = {'x1': data[1], 'x2': data[2],
               'x3': data[3], 'x4': data[4], 'x5': data[5]}
    return par_dict, ss_dict, data


def plot_time_perturbed_steady_state(ss_dict, tend=10_000, par='risk', rev_bool = False,
                                     random_bool=True, eps=0.01, par_dict=eq1_h1_par_dict, save_bool = False):
    """
    function to plot time series of the data
    :param par_val: parameter used for simulated bifurcation
    :param ss_dict: dictionrary of steady states for bifurcation
    :param data: data obtained from get_data()
    :param tend: final time for plot
    :param ode: ode geneeated from generate_ode()
    :param par: string used for the par_val
    :param rev_bool: boolean for using the revised model
    :param random_bool: boolean to use random initial conditions
    :param eps: small value used to perturb about the steady state
    :param par_dict: dictionary for parameters used for the bifurcation
    :return: pts -- pointset
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

    z, t = gen_sys(par_dict = par_dict, ics_dict = ss_dict_perturbed, tf = tend, rev_bool=rev_bool)
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
    if save_bool == True:
        plt.savefig(path + file_name, dpi=300)
    plt.show()

    """plt.plot(t, infected, 'r', label=r'$I$')
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
    plt.show()"""

    # plot the effective reproductive number
    if rev_bool == False:
        reff_num = par_dict['infection_bad'] * sb + (1 - par_dict['protection']) * par_dict['infection_good'] * v
        reff_den = par_dict['recovery'] + par_dict['education'] * good
        reff = reff_num / reff_den
        reff = 1 + (1 - par_dict['protection']) * par_dict['infection_good'] * v + par_dict['infection_bad'] * sb - par_dict['recovery'] -  par_dict['education'] * good
    else:
        reff_num = par_dict['infection_bad'] * sb + 0 * (1 - par_dict['protection']) * par_dict['infection_good'] * v
        reff_den = par_dict['recovery'] + par_dict['education'] * good
        reff = reff_num / reff_den
        reff = 1 + (1 - par_dict['protection']) * par_dict['infection_good'] * v + par_dict['infection_bad'] * sb - par_dict['recovery'] -  par_dict['education'] * good
    plt.plot(t, reff, 'b', lw = 2)
    plt.xlabel('t (Days)')
    plt.ylabel('$R_{eff}$')
    par_lab = par.title()
    file_name = f"\{par_lab}_reff.jpeg"
    if save_bool == True:
        plt.savefig(path + file_name, dpi=300)
    plt.show()

    return pts




def generate_1d_bifurcation(ode = '', par = 'risk', par_d = eq1_h1_par_dict, ics_d = eq1_h1_ss, tend = 1_000, type = 'EP-C', max_step = None, min_step = None, step = None, max_points = None, forward_bool = True, backward_bool = True, rev_bool = False):
    """
    function to generate a 1 dimension bifurcation...
    :param par: parameter used for bifurcation: risk, protection, infection_good, infection_bad, misinformation, education, recovery
    :param par_d: dictionary for parameters used for time simulation and initialization
    :param ics_d: dictionary for initial conditions for the time simisinformationlation
    :param tend: final time of simulation
    :param type: LP-C, FP-C, etc.
    :param max_step: largest step size of bifurcation
    :param min_step: smallest step size of bifurcation
    :param step_size: size of the step for the bifurcation
    :param max_points: total number of points max for the bifurcation
    :param forward_bool: Boolean to compute forward
    :param backward_bool: Boolean to compute backward
    :param rev_bool: boolean for the revised model (short term == True)
    :return: PC -- continuation curve for the bifurcation analysis
    """
    # generate an ode
    #ode = generate_ode(p_d = par_d, i_d = ics_d, tend=tend, rev_bool = rev_bool)

    # pts, ss_dict = generate_pointset(ode)

    # use the initial conditions at the steady state
    # ode.set(ics = ss_dict)

    # ----------------------------- Initialization ---------

    # determine the label of the parameter:
    if par == 'risk':
        par_lab = r'$r$'
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-4
        if max_points == None:
            max_points = 750
    elif par == 'education':
        par_lab = r'$\epsilon$'
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-7
        if step == None:
            step = 1e-4
        if max_points == None:
            max_points = 300
    elif par == 'misinformation':
        par_lab = r'$\mu$'
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-3
        if max_points == None:
            max_points = 200
    elif par == 'infection_good':
        par_lab = r'$\chi$'
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-3
        if max_points == None:
            max_points = 200
    elif par == 'infection_bad':
        par_lab = r'$\hat{\chi}$'
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-3
        if max_points == None:
            max_points = 200
    elif par == 'recovery':
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-3
        if max_points == None:
            max_points = 200
    elif par == 'protection':
        if max_step == None:
            max_step = 1e-2
        if min_step == None:
            min_step = 1e-5
        if step == None:
            step = 1e-3
        if max_points == None:
            max_points = 200
    else:
        print('Choose one of the following:')
        print('protection, risk, education, misinformation, infection_good, infection_bad, recovery')
        quit()

    curve = f"EQ{par}"

    # generate an ode
    if ode == '':
        ode = generate_ode(par_d=par_d, ics_d=ics_d, tend=500, rev_bool=rev_bool)
    pts = ode.compute('polarize').sample(dt = 1)
    ss_dict = {'x1': pts['x1'][-1], 'x2': pts['x2'][-1], 'x3': pts['x3'][-1],
                   'x4': pts['x4'][-1], 'x5': pts['x5'][-1]}
    #ode.set(ics = ss_dict)

    # generate a continuation curve
    PC = ContClass(ode)
    PCargs = dst.args(name=curve, type=type)
    PCargs.freepars = [par]  # should be one of the parameters from DSargs.pars --

    # change the default settings for the numerical continuation
    PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
    # choose carefully
    PCargs.MaxStepSize = max_step # greatest step size of the bifurcation
    PCargs.MinStepSize = min_step # smallest step size of the bifurcation
    PCargs.StepSize = step # step size of the bifurcation
    PCargs.LocBifPoints = 'all'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!
    PCargs.verbosity = 2 # console detail display
    PCargs.StopAtPoints = 'B' # boundary point
    PC.newCurve(PCargs) # continuation for the curve

    # continue backwards and forwards!
    if forward_bool == True:
        PC[curve].forward()
    if backward_bool == True:
        PC[curve].backward()



    if par == 'risk':
        PCargs.name = 'EQrisk2' # name of the additional curve
        PCargs.type = 'EP-C' # equilibrium point curve
        PCargs.initpoint = {'x1': 0.0005714445346933421,
                            'x2': 0.1893893682498801,
                            'x3': 0.1968691906600376,
                            'x4': 0.6047210552785887,
                            'x5': 1.0,
                            'risk': 0.0}

        PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars
        # change the default settings for the numerical continuation
        PCargs.MaxNumPoints = int(1.0 * max_points)  # The following 3 parameters are set after trial-and-error
        PCargs.MaxStepSize = 0.01 # max step size
        PCargs.MinStepSize = 1e-5 # min step size
        PCargs.StepSize = 0.1e-3 # step size
        PCargs.StopAtPoints = 'B' # boundary point
        PCargs.LocBifPoints = 'BP'  # detect limit points / saddle-node bifurcations
        PCargs.SaveEigen = True  # to tell unstable from stable branches
        PCargs.SaveJacobian = True  # saves the Jacobian data which can be used for the nullclines!

        # PC.update(PCargs)
        PC.newCurve(PCargs)
        PC['EQrisk2'].forward()

    elif par == 'protection':
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


    # --------------------------------- Plotting Time -----------
    # begin plotting
    yvar_array = ['x4', 'x3', 'x2', 'x1', 'x5']
    ylab_array = [r'$V$', r'$I_B$', r'$S_B$', r'$S_G$', r'$\phi$']
    col_array = ['g', 'r', 'r', 'b', 'k']
    for z, z0 in enumerate(yvar_array):
        # display the bifurcation!
        PC.display([par, yvar_array[z]], stability=True, color=col_array[z])  # x variable vs y variable!
        # disable the boundary
        PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='P')
        # PC.plot.toggleLabels(visible='off', bylabel=None, byname=None, bytype='B')
        plt.title('')  # no title
        plt.ylabel(ylab_array[z])
        plt.xlabel(par_lab)
        plt.show()
    return PC


def sample_new(par_dict = eq1_h1_par_dict, ics_dict = eq1_h1_ss, tf = 10_000, nullcline_bool = True, time_bool = True, bifurcation_bool = True, par = 'risk', special_point = 'H1', eps0 = 0.10, min_step = None, max_step = None, step = None, max_points = None, user_input_bool = False, rev_bool = False):
    """
    function to generate a plotted simulation of the sample
    :param par_dict: dictionary of parameters
    :param ics_dict: dictionary of initial conditions
    :param tf: final time of the simulation
    :param nullcline_bool: boolean to plot the nullclines
    :param time_bool: boolean to plot in the time simulation
    :param bifurcation_bool: boolean to plot the bifurcation
    :param par: parameter used for the bifurcation: risk, protection, infection_good, infection_bad, recovery, education, misinformation
    :param special_point: parameter used for the special point of the bifurcation
    :param eps0: weight of the perturbed data of the special point obtained in the bifurcation
    :param min_step: smallest step of the bifurcation
    :param max_step: largest step of the bifurcation
    :param step: initial step size of the bifurcation
    :param max_points: maximum number of points in the bifurcation
    :param rev_bool: boolean -- True for the short run model -- False for the long run model
    :return: plotted simulation of the bifurcation
    """
    # now we generate the ode!
    z, t = gen_sys(par_dict = par_dict, ics_dict = ics_dict, tf = tf, rev_bool = rev_bool)
    # generate plot time series
    ss_d = plot_time(z, t, par_dict = par_dict)
    # use ss_d to generate an ode in the short time
    ode = generate_ode(par_d = par_dict, ics_d = ics_dict, tend = int(tf * 0.01), rev_bool = rev_bool)
    # generate a bifurcation
    if bifurcation_bool == True:
        # perform a continuation in 1d for the selected parameter
        PC = generate_1d_bifurcation(ode = ode, par=par, par_d=par_dict, ics_d=ics_dict, tend=tf, type='EP-C',
                                    max_step=max_step, min_step=min_step, step=step, max_points=max_points, forward_bool=True,
                                    backward_bool=True, rev_bool=rev_bool)
        # get the continuation data about the special point
        flag = 1
        while flag != 0:
            if user_input_bool == False:
                spec_point = special_point
            else:
                print(PC.info())
                #curve = input('Choose one of the curves above^^...')
                #print(PC[f"EQ{par}"].sol()) # printing the data associated to the continuation curve....
                spec_point = input('Choose one of the special points from above of the plotted bifurcation...   "0" to stop')

            if spec_point == 'BP1':
                par_dict_new, ss_dict_new, data_new = get_data(PC, curve=f"EQ{par}2", special_point=spec_point, par=par, par_dict=par_dict)
            else:
                par_dict_new, ss_dict_new, data_new = get_data(PC, curve=f"EQ{par}", special_point=spec_point, par=par, par_dict=par_dict)

            # generate a purturbance, 'delta' about the special point
            if time_bool == True:
                if user_input_bool == True:
                    eps0 = float(input('Choose a value between 0 and 1 for perturbation...'))
                else:
                    eps0 = eps0
                pts_delta = plot_time_perturbed_steady_state(ss_dict_new, tend=tf, par=par, rev_bool = rev_bool,
                                                 random_bool=True, eps=eps0, par_dict=par_dict_new)

                if nullcline_bool == True:
                    option_array = ['A', 'B', 'C']
                    # iterate over the arrays to obtain nice nullcline figures....
                    for i, i0 in enumerate(option_array):
                        if rev_bool == False:
                            plot_nullclines(option=i0, PTS=pts_delta, par_dict=par_dict_new, ss_dict=ss_dict_new, n_bin=500, xlow=None,
                                            xhigh=None, ylow=None, yhigh=None, quiv_bool=True, w0=0, distance=0.05, z0=0,
                                            evecs_bool=False, evecs=None, ics_dict={}, par=par, seq='', title_bool=False, title=spec_point)
                        else:
                            plot_nullclines_rev(option=i0, PTS=pts_delta, par_dict=par_dict_new, ss_dict=ss_dict_new,
                                            n_bin=500, xlow=None,
                                            xhigh=None, ylow=None, yhigh=None, quiv_bool=True, w0=0, distance=0.05,
                                            z0=0,
                                            evecs_bool=False, evecs=None, ics_dict={}, par=par, seq='',
                                            title_bool=False, title=spec_point)
                    if user_input_bool == False:
                        flag = 0
                        break
        return par_dict_new, ss_dict_new, data_new


par_dict_new, ss_dict_new, data_new = sample_new(par = 'risk', special_point = 'H1', max_step = 0.01, min_step = None, step = None, max_points = 500, eps0 = 0.10, user_input_bool = False)
#par_dict_new_rev, ss_dict_new_rev, data_new_rev = sample_new(par = 'risk', tf = 5000, special_point = 'H1', ics_dict = ss_dict_new, par_dict = par_dict_new, max_step = 0.05, min_step = None, step = None, max_points = 500, eps0 = 0.05, user_input_bool = False, rev_bool = True)
print(par_dict_new, ss_dict_new, data_new)
par_dict_new, ss_dict_new, data_new = sample_new(par = 'misinformation', par_dict = par_dict_new, ics_dict = ss_dict_new, special_point = 'H1', max_step = 0.01, min_step = None, step = None, max_points = 300, eps0 = 0.05, user_input_bool = True, rev_bool = False)
#par_dict_new, ss_dict_new, data_new = sample_new(par = 'education', special_point = 'LP1', max_step = 0.005, min_step = None, step = None, max_points = None)
#par_dict_new, ss_dict_new, data_new = sample_new(par = 'protection', special_point = 'H1', max_step = 0.005, min_step = None, step = None, max_points = None)