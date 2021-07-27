import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
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
    for z in range(0, len(row_label)):
        if z % 4 != 0:
            print(z % 4)
            row_label[z] = ''
            col_label[z] = ''

    print(col_label)

    df = pd.DataFrame(array, columns = col_label, index = row_label)



    s = sns.heatmap(df, cmap = cmap_bin['Blues'])


    z = 52 % 4
    print(z)

    s.set_xticklabels(col_label)
    s.set_yticklabels(row_label)

    plt.xlabel('$r$', fontsize = 18)
    plt.ylabel('$\gamma$', fontsize = 18)

    if show_bool == True:
        file_name = save_path + save_name
        plt.savefig(file_name, dpi = 300)
        plt.show()
