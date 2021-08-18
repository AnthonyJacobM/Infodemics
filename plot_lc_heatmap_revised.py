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
        xlab = r'$\tilde{\chi}_{gb}$'
    elif xpar == 'misinformation':
        xlab = r'$\tilde{\chi}_{bg}$'
    elif xpar == 'infection_good':
        xlab = r'$\chi_{gb}$'
    elif xpar == 'infection_bad':
        xlab = r'$\hat{\chi}_{bb}$'
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
        ylab = r'$\tilde{\chi}_{gb}$'
    elif ypar == 'misinformation':
        ylab = r'$\tilde{\chi}_{bg}$'
    elif ypar == 'infection_good':
        ylab = r'$\chi_{gb}$'
    elif ypar == 'infection_bad':
        ylab = r'$\hat{\chi}_{bb}$'
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


"""p0 = 'education'

file_vac = f"risk_{p0}_vac.txt"
file_bad = f"risk_{p0}_bad.txt"
file_inf = f"risk_{p0}_inf.txt"
#file_xlimit = f"lc_risk_{p0}_xlimit.txt"
#file_ylimit = f"lc_risk_{p0}_ylimit.txt"


data_vac = np.loadtxt(file_vac, delimiter = ',', dtype = float)
data_bad = np.loadtxt(file_bad, delimiter = ',', dtype = float)
data_inf = np.loadtxt(file_inf, delimiter = ',', dtype = float)

#xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
#ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

lc_x, lc_y = plot_lc_heatmap(data=data_bad, zvar='bad', xpar='risk', ypar=p0, cmap='Greys', ylow=0.05, yhigh=0.5, xlow = 1.0, xhigh = 1.9)

plt.plot(lc_x, lc_y, 'b', lw = 5)
plt.fill(lc_x, lc_y, 'k')
plt.show()"""