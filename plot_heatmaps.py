import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
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

par_x_bin = np.linspace(xlow, xhigh, 100)
par_y_bin = np.linspace(ylow, yhigh, 100)
row_array = [str(np.round(x0, 3)) for x, x0 in enumerate(par_x_bin)]
col_array = [str(np.round(x0, 3)) for x, x0 in enumerate(par_y_bin)]


def plot_heatmaps(save_path = path, save_name = '\heatmap.jpg', data = array, xlabel = '$r', ylabel = '$\tilde{\chi}_{bg}$', col_labels = row_array, row_labels = col_array, cmap = cmap_bin['hot'], show_bool = True):
    """
    function to generte heatmaps with user input data
    :param save_path: path to save the figure
    :param save_name: file name unique to the figure
    :param data: user input numpy data
    :param xlabel: label on the x axis
    :param ylabel: label on the y axis
    :param col_labels: numbers for the label on the heatmap (xpar)
    :param row_labels: numbers for the label on the heatmap (ypar)
    :param cmap: color used for the heatmap; use one from the dictionary
    :param show_bool: boolean to show the figure
    :return: plotted heatmap
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    array = np.random.random((100,100))
    beta_uhat_bin = np.linspace(0, 1, 100)
    col_label = [str(np.round(x,3)) for v, x in enumerate(beta_uhat_bin)]
    row_label = [str(np.round(x,3)) for v, x in enumerate(beta_uhat_bin)]

    # generate spaces for col_label and row_label
    """for z in range(0, len(row_label)):
        if z % 4 != 0:
            print(z % 4)
            row_label[z] = ''
            col_label[z] = ''"""



    df = pd.DataFrame(array, columns = col_label, index = row_label)



    s = sns.heatmap(df, cmap = cmap_bin['Blues'])
    ax.set_xticklabels(col_label)
    ax.set_yticklabels(row_label)

    # get the axes for the linear spaced points in x and y!
    xbin = ax.get_xticklabels()
    ybin = ax.get_yticklabels()
    xticks, xlabels = plt.xticks()
    yticks, ylabels = plt.yticks()

    # generate new xlabels and ylabels
    lb_x = 0
    ub_x = 1
    N_x = len(xlabels)
    lb_y = 0
    ub_y = 1
    N_y = len(ylabels)
    x_bin = np.linspace(lb_x, ub_x, N_x)
    y_bin = np.linspace(lb_y, ub_y, N_y)
    xlabels_new = [str(np.round(x0,2)) for x, x0 in enumerate(x_bin)]
    ylabels_new = [str(np.round(y0, 2)) for y, y0 in enumerate(y_bin)]

    for x, x0 in enumerate(xlabels):
        if x % 2 == 0:
            xlabels[x] = xlabels_new[x]
        else:
            xlabels[x] = ''

    for y, y0 in enumerate(ylabels):
        if y % 2 == 0:
            ylabels[y] = ylabels_new[y]
        else:
            ylabels[y] = ''

    # use new labels!
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)


    plt.xlabel('$r$', fontsize = 18)
    plt.ylabel('$\gamma$', fontsize = 18)

    if show_bool == True:
        file_name = save_path + save_name
        plt.savefig(file_name, dpi = 300)
        plt.show()


def example_heatmap():
    """
    function to plot a heatmap easily!
    :return: plotted heatmap!
    """
    # using a function to plot a heatmap!
    plot_heatmaps()

example_heatmap()
