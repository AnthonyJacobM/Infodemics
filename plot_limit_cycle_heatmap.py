import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# change default mpl settings
# get a path associated to saving each figure
path = r'D:\Users\antho\PycharmProjects\Infodemics\figures'
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

    xlimit = np.loadtxt(file_xlimit, delimiter=',', dtype=float)
    ylimit = np.loadtxt(file_ylimit, delimiter=',', dtype=float)

    n = len(data)
    par_x_bin = np.linspace(xlimit[0], xlimit[-1], n)
    par_y_bin = np.linspace(ylimit[0], ylimit[-1], n)
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
    im = ax.imshow(data, cmap = cmap, extent = [xlimit[0], xlimit[-1], ylimit[0], ylimit[-1]])

    #plt.xlim(xlimit[0], xlimit[-1])
    #plt.ylim(ylimit[0], ylimit[-1])

    plt.xlabel(xlab, fontsize = 18)
    plt.ylabel(ylab, fontsize = 18)




    if show_bool == True:
        file_name = save_path + save_name
        plt.savefig(file_name, dpi = 300)
        plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8), dpi=100, subplot_kw={'aspect': 'equal'})

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

    ax1.yaxis.set_tick_params(labelsize='xx-large')
    ax1.xaxis.set_tick_params(labelsize='xx-large')


    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_lab, fontsize = 18)
    ax1.plot(lc_x, lc_y, color = lc_col, lw = 5)
    ax1.fill(lc_x, lc_y, col = 'k')
    plt.show()

    plt.close(fig)

    return lc_x, lc_y


"""p0 = 'misinformation'

file_vac = f"risk_{p0}_vac.txt"
file_bad = f"risk_{p0}_bad.txt"
file_inf = f"risk_{p0}_inf.txt"
file_xlimit = f"lc_risk_{p0}_xlimit.txt"
file_ylimit = f"lc_risk_{p0}_ylimit.txt"


data_vac = np.loadtxt(file_vac, delimiter = ',', dtype = float)
data_bad = np.loadtxt(file_bad, delimiter = ',', dtype = float)
data_inf = np.loadtxt(file_inf, delimiter = ',', dtype = float)
xlimit = np.loadtxt(file_xlimit, delimiter = ',', dtype = float)
ylimit = np.loadtxt(file_ylimit, delimiter = ',', dtype = float)

lc_x, lc_y = plot_limit_cycle_heatmap(data=data_vac, zvar='inf', xpar='risk', ypar=p0, cmap='Reds')


plt.plot(lc_x, lc_y, 'b', lw = 5)
plt.fill(lc_x, lc_y, 'b')
plt.show()"""