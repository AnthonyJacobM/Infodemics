import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gen_sys import gen_sys
from generate_ode import generate_ode
from generate_pointset import generate_pointset
from PyDSTool import *
from scipy.signal import argrelextrema
import PyDSTool as dst
from perturb_ics import perturb_ics
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_row = np.linspace(0.50, 1, 10)
def plot_colorbar(N = 10, CMAP = 'RdBu', cbar_lab = r'$\delta$', data_row = data_row):

    # generate new empty arrange
    fig, ax1 = plt.subplots(1,1)
    nbin = N
    cmap = CMAP
    cbarlab = cbar_lab
    cbar_lab = cbarlab
    data = np.zeros((nbin, nbin))

    for i in range(nbin):
        print(data[i])
        data[i] = data_row

    im = ax1.imshow(data, cmap = cmap)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="8.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_lab, fontsize = 18)
    plt.show()
    plt.close(fig)
    plt.show()


# key vars used for simulation
k_vars = ['x1', 'x2', 'x3', 'x4', 'x5']

cmap = plt.get_cmap('RdBu')

# initialize steady state at H1 of Risk
ss_ics = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

# intialize parameters at H1 of Risk
par_ics = {'recovery': 0.07, 'belief': 1.0,
            'risk': 1.635, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}



# generate the ode system
ode = generate_ode(par_dict = par_ics, ics_dict = ss_ics, tf = 100)

# generate a pointset
pts, ss_dict = generate_pointset(ode = ode)


# obtain the steady state values of time series


# generate a bifurcation and obtain the Hopf

ode.set(ics = ss_dict)
max_points = 200

# generate a continuation curve
PC = ContClass(ode)
PCargs = dst.args(name = 'EQ', type = 'EP-C')
PCargs.freepars = ['risk']  # should be one of the parameters from DSargs.pars --

# change the default settings for the numerical continuation
PCargs.MaxNumPoints = max_points  # The following 3 parameters are set after trial-and-error
# choose carefully
PCargs.MaxStepSize = 0.1
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
PC['EQ'].forward()
PC['EQ'].backward()

# display the curve
PC['EQ'].display(['risk', 'x3'], color = 'black', stability = True)
plt.xlabel('$r$', fontsize = 18)
plt.ylabel(r'$\phi$', fontsize = 18)
plt.title('')
plt.show()

# obtain the Hopf bifurcation
H1 = PC['EQ'].getSpecialPoint('H1')

# start a new bifurcation plot
# update at the new risk for H1
par_ics['risk'] = H1['risk']

# update the steady state values at the H1
for k, k0 in enumerate(k_vars):
    ss_ics[k0] = H1[k0]

# update the ode
ode.set(pars = par_ics, ics = ss_ics)




def sample():
    """
    function to export the data to an excel file
    :return:
    """
    # chaos in education
    par_royce_request = {'recovery': 0.07, 'belief': 1.0, 'risk': 1.635295791362042, 'protection': 0.9, 'education': 0.3886273053001566, 'misinformation': 0.0660611192767927, 'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}
    ics_royce_request =  {'x1': 0.1749602436240004, 'x2': 0.2532926229438787, 'x3': 0.24075044102064777, 'x4': 0.2169574798040472, 'x5': 0.0014329913913326327}

    par_royce_request['protection'] = 1
    ics_royce_request['x5'] = 1e-5
    ics_royce_request['x4'] = 0.15


    # generate ode
    z, t = gen_sys(par_dict = par_royce_request, ics_dict=ics_royce_request, tf = 5_000)
    x1, x2, x3, x4, x5 = z.T
    plt.plot(t, x1, 'b', label = 'SG')
    plt.plot(t, x2, 'r', label = 'SB')
    plt.plot(t, x3, 'k', label = 'IB')
    plt.plot(t, x4, 'b--', label = 'V')
    plt.plot(t, x5, 'm', label = r'$\phi$')
    plt.legend()
    plt.xlabel('t')
    plt.show()

    dict_excel = {'sg': x1,
                  'sb': x2,
                  'ib': x3,
                  'v': x4,
                  'phi': x5,
                  'ig': 1 - (x1 + x2 + x3 + x4)}

    par_excel = {'gamma': 0.07,
                 'm': 1.0,
                 'r': 1.635295791362042,
                 'delta': 1.0,
                 'epsilon': 0.3886273053001566,
                 'mu': 0.0660611192767927,
                 'chi': 0.048,
                 'chi_hat': 0.37}

    df_time_series = pd.DataFrame(data = dict_excel, columns = ['sg', 'sb', 'ib', 'v', 'phi', 'ig'])
    df_par = pd.DataFrame(data = par_excel, columns = ['gamma', 'm', 'r', 'delta', 'epsilon', 'mu', 'chi', 'chi_hat'], index = [0.07, 1.0, 1.635295791362042, 1.0, 0.3886273053001566, 0.0660611192767927, 0.048, 0.37])
    df_par = pd.DataFrame(data = par_excel, columns = ['gamma', 'm', 'r', 'delta', 'epsilon', 'mu', 'chi', 'chi_hat'], index = [0])
    df_time_series.to_excel('royce_request_time_series.xlsx', index = True)
    df_par.to_excel('royce_request_par.xlsx', index = True)





def generate_risk_time_series(mc_z = 10, epsilon = 0.01, n0 = 51, tend = 11_000, zvar = 'bad', leg_bool = False):
    """
    function to generate a time series in risk
    :param mc_z: monte carlo in z
    :param epsilon: weight of the noise
    :param n0: size of array
    :param tend: final time of numerical solution
    :param zvar: variable along the z axis
    :return: plotted solution
    """
    # now we generate the data
    # now we generate period and amplitude data for x variables
    r_low = 0
    r_high = 2.5
    n = int(n0)
    r_bin = np.linspace(r_low, r_high, n)

    rnew_bin = []

    # empty bin
    max_inf_peak_bin = []
    max_bad_peak_bin = []
    mean_inf_peak_bin = []
    mean_bad_peak_bin = []
    period_bin = []

    z_mc = mc_z # number mc bool
    z, t = gen_sys(par_dict=par_ics, ics_dict=ss_ics, tf=10_000)
    N = len(t)

    for i, i0 in enumerate(r_bin):
        rnew = np.round(i0, 3)

        # change parameter
        par_ics['risk'] = rnew

        # MC LOOP
        inf0 = np.zeros((z_mc, N)) # empty for mc average
        bad0 = np.zeros((z_mc, N)) # empty for mc average
        phi0 = np.zeros((z_mc, N))
        good0 = np.zeros((z_mc, N))
        vac0 = np.zeros((z_mc, N))

        for w in range(z_mc):
            # perturb the steady state
            ics_new = perturb_ics(ics_dict=ss_ics, eps0 = epsilon)
            # simulate ode, get state variables
            z, t = gen_sys(par_dict = par_ics, ics_dict = ics_new, tf = tend)

            # unpack
            sg, sb, ib, v, phi = z.T
            # generate auxillary mc functions
            inf0[w][:] = 1 - (sg + sb + v)
            bad0[w][:] = sg + sb
            phi0[w][:] = phi
            good0[w][:] = 1 - (sg + sb)
            vac0[w][:] = v


        # mc weighted average
        inf = np.mean(inf0, axis = 0)
        bad = np.mean(bad0, axis = 0)
        phi = np.mean(phi0, axis = 0)
        vac = np.mean(vac0, axis = 0)
        good = np.mean(good0, axis = 0)

        # determine the label for the axis of the plot
        if zvar == 'inf':
            ylab = r'$Infected$'
            ybin = inf
        elif zvar == 'bad':
            ylab = r'$Bad$'
            ybin = bad
        elif zvar == 'good':
            ylab = r'$Good$'
            ybin = good
        elif zvar == 'vac':
            ylab = r'$Vaccinated$'
            ybin = vac
        else:
            ylab = r'$\phi$'
            ybin = phi

        # get infected / bad peak
        inf_peak = inf[argrelextrema(inf, np.greater)[0]]
        bad_peak = bad[argrelextrema(bad, np.greater)[0]]
        idx_bin = argrelextrema(inf, np.greater)[0]



        if np.max(inf) <= 0.8 and np.min(inf) >= 0 and np.max(bad) <= 1 and np.min(bad) >= 0 and np.max(phi) <= 1.01:
            col = cmap(float(i / n))
            print(i)
            plt.plot(ybin[1000:], 'r-', color = col, lw = 2)

            if len(inf_peak) > 0:
                max_inf_peak_bin.append(np.max(inf_peak))
                mean_inf_peak_bin.append(np.average(inf_peak))
                period_bin.append(np.average(idx_bin[1:-1] - idx_bin[0:-2]))

            else:
                max_inf_peak_bin.append(np.max(inf))
                mean_inf_peak_bin.append(np.average(inf))
                period_bin.append(0)

            if len(bad_peak) > 0:
                max_bad_peak_bin.append(np.max(bad_peak))
                mean_bad_peak_bin.append(np.average(bad_peak))

            else:
                max_bad_peak_bin.append(np.max(bad))
                mean_bad_peak_bin.append(np.average(bad))

            rnew_bin.append(rnew)

            if i == 5 or i == 18 or i == 45 or i == 30:
                plt.plot(ybin[1000:], 'r-', color=col, lw=5, label=f"r = {rnew}")



    plt.xlabel('t', fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)
    #plt.ylim([0, 0.5])
    if leg_bool == True:
        plt.legend()
    plt.show()

    """# save the data
    np.savetxt('max_inf_peak_risk.txt', max_inf_peak_bin, delimiter = ',')
    np.savetxt('max_bad_peak_risk.txt', max_bad_peak_bin, delimiter = ',')

    np.savetxt('mean_inf_peak_risk.txt', mean_inf_peak_bin, delimiter = ',')
    np.savetxt('mean_bad_peak_risk.txt', mean_bad_peak_bin, delimiter = ',')

    np.savetxt('period_risk.txt', period_bin, delimiter = ',')"""


    """# plot the data
    plt.plot(rnew_bin, max_inf_peak_bin, 'r', fillstyle = 'full', ms = 10)
    plt.xlabel(r'$r$', fontsize = 18)
    plt.ylabel('Max Peak Infected', fontsize = 18)
    plt.show()

    plt.plot(rnew_bin, max_bad_peak_bin, 'r', fillstyle = 'full', ms = 10)
    plt.xlabel(r'$r$', fontsize = 18)
    plt.ylabel('Max Peak Bad', fontsize = 18)
    plt.show()

    plt.plot(rnew_bin, mean_inf_peak_bin, 'r', fillstyle = 'full', ms = 10)
    plt.xlabel(r'$r$', fontsize = 18)
    plt.ylabel('Mean Peak Bin', fontsize = 18)
    plt.show()

    plt.plot(rnew_bin, mean_bad_peak_bin, 'k', fillstyle = 'full', ms = 10)
    plt.xlabel(r'$r$', fontsize = 18)
    plt.ylabel('Mean Peak Bad', fontsize = 18)
    plt.show()"""

    plt.plot(rnew_bin, period_bin, 'b', fillstyle = 'full', ms = 10)
    plt.xlabel(r'$r$', fontsize = 18)
    plt.ylabel('Period', fontsize = 18)
    plt.show()



def plot_risk():
    """
    function to plot education
    :return:
    """
    zvar_bin = ['inf', 'bad', 'good', 'vac', 'phi']
    for z, z0 in enumerate(zvar_bin):
        generate_risk_time_series(zvar = z0, n0 = 101, mc_z = 10, tend = 10_000)

#plot_risk()
data_row = np.linspace(0, 2.5, 10)
plot_colorbar(N = 10, CMAP = 'RdBu', cbar_lab = r'$r$', data_row = data_row)



# try to generate a 2D heatmap should follow a similar coding setup to the previous plotting time seriees
def generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 11, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 2.5, y_low = 0, y_high = 0.40, x_par = 'risk', y_par = 'misinformation', save_bool = True):
    """
    function to generate the data of the two dimensional heatmap for $r$
    :param mc_z: number of MCMC simulations
    :param epsilon: weight of the purturbation
    :param n0: size of the 2d array : n0 x n0
    :param tend: final time of the simulation
    :param zvar: variable to be plotted: 'bad', 'inf', 'vac', 'phi', 'good'
    :return: generated heatmap
    :param x_low: LB on the x parameter
    :param x_high: UB on the x parameter
    :param y_low: LB on the y parameter
    :param y_high: UB on the y parameter
    :param x_par: parameter for the x-bin
    :param y_par: parameter for the y-bin
    :param save_bool: boolean to save the generated data
    :return: n0 x n0 generated heatmap
    """
    # -- begin!
    inf_peak_bin = []
    bad_peak_bin = []
    vac_peak_bin = []
    period_bin = []
    x_bin = np.linspace(x_low, x_high, n0) # x array
    y_bin = np.linspace(y_low, y_high, n0) # y array

    # re-name it!
    z_mc = mc_z
    # intialize the system!
    z, t = gen_sys(par_dict = par_ics, ics_dict = ss_ics, tf = tend)
    N = len(t) # number of data points in simulation of each state variable

    # iterate over the x bin
    for x, x0 in enumerate(x_bin):
        # rename the x parameter
        par_ics[x_par] = np.round(x0, 3)


        # iterate over the y bin
        for y, y0 in enumerate(y_bin):
            # rename the y parameter
            par_ics[y_par] = np.round(y0, 3)

            print(x, 'out of ', n0, y, 'out of', n0)

            # generate an empty for the MCMC average
            inf0 = np.zeros((z_mc, N))
            bad0 = np.zeros((z_mc, N))
            phi0 = np.zeros((z_mc, N))
            vac0 = np.zeros((z_mc, N))

            # append x and y parameter values
            # MC loop  --- time for the average over the initial conditions
            for w in range(z_mc):
                # generate a purturbation about the random initial condition
                ics_new = perturb_ics(ics_dict = ss_dict, eps0 = epsilon)

                # simulate the ode, obtain state variables
                z, t = gen_sys(par_dict = par_ics, ics_dict = ics_new, tf = tend)

                # unpack it!
                sg, sb, ib, v, phi = z.T
                # generate the auxillary functions
                inf0[w][:] = 1 - (sg + sb + v)
                bad0[w][:] = sg + sb
                phi0[w][:] = phi
                vac0[w][:] = v

            # generate the MCMC averages!
            inf = np.mean(inf0, axis = 0)
            bad = np.mean(bad0, axis = 0)
            phi = np.mean(phi0, axis = 0)
            vac = np.mean(vac0, axis = 0)

            # get the peaks
            inf_peak = inf[argrelextrema(inf, np.greater)[0]]
            bad_peak = bad[argrelextrema(bad, np.greater)[0]]
            idx_peak = argrelextrema(inf, np.greater)[0]
            vac_peak = vac[argrelextrema(vac, np.greater)[0]]

            if np.max(inf) <= 0.8 and np.min(inf) >= 0 and np.max(bad) <= 1 and np.min(bad) >= 0 and np.max(
                    phi) <= 1.01:

                if len(inf_peak) > 0:
                    inf_peak_bin.append(np.average(inf_peak))
                    period_bin.append(np.average(idx_peak[1:-1] - idx_peak[0:-2]))
                else:
                    inf_peak_bin.append(np.average(inf))
                    period_bin.append(0)
                if len(bad_peak) > 0:
                    bad_peak_bin.append(np.average(bad_peak))
                else:
                    bad_peak_bin.append(np.average(bad))
                if len(vac_peak) > 0:
                    vac_peak_bin.append(np.average(vac_peak))
                else:
                    vac_peak_bin.append(np.average(vac))


    if save_bool == True:
        # save the data -
        np.savetxt(f"inf_peak_{x_par}_{y_par}_new_.txt", inf_peak_bin, delimiter=',')
        np.savetxt(f"bad_peak_{x_par}_{y_par}_new_.txt", bad_peak_bin, delimiter=',')
        np.savetxt(f"vac_peak_{x_par}_{y_par}_new_.txt", vac_peak_bin, delimiter=',')
        np.savetxt(f"period_{x_par}_{y_par}_new_.txt", period_bin, delimiter=',')
        np.savetxt(f"x_bin_{x_par}_{y_par}_new_.txt", x_bin, delimiter = ',')
        np.savetxt(f"y_bin_{x_par}_{y_par}_new_.txt", y_bin, delimiter = ',')





generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 2.5, y_low = 0, y_high = 0.50, x_par = 'risk', y_par = 'misinformation', save_bool = True)
generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 0.4, y_low = 0, y_high = 0.50, x_par = 'education', y_par = 'misinformation', save_bool = True)
generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 0.4, y_low = 0, y_high = 0.50, x_par = 'protection', y_par = 'misinformation', save_bool = True)


generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 2.5, y_low = 0, y_high = 0.40, x_par = 'risk', y_par = 'education', save_bool = True)
generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0.5, x_high = 1.0, y_low = 0, y_high = 0.40, x_par = 'protection', y_par = 'education', save_bool = True)

generate_2D_heatmap(mc_z = 10, epsilon  = 0.01, n0 = 101, tend = 10_000, zvar = 'inf', x_low = 0, x_high = 2.5, y_low = 0.50, y_high = 1, x_par = 'risk', y_par = 'protection', save_bool = True)