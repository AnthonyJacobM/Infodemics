import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# reference:
# https://stackoverflow.com/questions/32132773/a-tornado-chart-and-p10-p90-in-python-matplotlib

def plot_tornado(state_var = 'V_ss', col = 'blue'):
    """
    function to plot tornado heatmaps
    :param state_var: V_ss or I_ss or R_e
    :param col: color of the plot
    :return: plotted tornado figure
    """
    # begin
    # variables of the system
    variables = ['$\hat{\chi}$', r'$\mu$', r'$\epsilon$', r'$\gamma$', r'$r$', r'$m$', r'$\delta$']

    # middle axis
    base = 0

    # vaccinated
    lows_vac = np.array([0.0, 0, -0.14, -0.76, -1.4, 0, 0])
    values_vac = np.array([0.038, 0.11, 0, 0, 0, 0.21, 0.23])

    # infected
    lows_inf = np.array([-0.59, 0, -0.67, 0, 0, -0.011])
    values_inf = np.array([0, 0.67, 0.61, 0, 0.4, 0.36, 0])

    # Reff
    lows_reff = np.array([0, 0, -0.82, -0.18, 0, 0, 0])
    values_reff = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    if state_var == 'V_ss':
        lows = lows_vac
        values = values_vac
        title_string = r'$V_{ss}$'
    elif state_var == 'I_ss':
        lows = lows_inf
        values = values_inf
        title_string = r'$I_{ss}$'
    elif state_var == 'Re':
        lows = lows_reff
        values = values_reff
        title_string = r'$R_e$'
    else:
        print('Choose: V_ss or I_ss or Re')
        quit()


    ###############################################################################
    # The actual drawing part

    # The y position for each variable
    ys = range(len(values))[::-1]  # top to bottom

    # Plot the bars, one by one
    for y, low, value in zip(ys, lows, values):
        # The width of the 'low' and 'high' pieces
        low_width = base - low
        high_width = low + value - base

        # Each bar is a "broken" horizontal bar chart
        plt.broken_barh(
            [(low, low_width), (base, high_width)],
            (y - 0.4, 0.8),
            facecolors=['red', col],  # Try different colors if you like
            edgecolors=['black', 'black'],
            linewidth=1,
        )

        # Display the value as text. It should be positioned in the center of
        # the 'high' bar, except if there isn't any room there, then it should be
        # next to bar instead.
        x = base + high_width / 2
        if x <= base + 50:
            x = base + high_width + 50
        plt.text(x, y, str(value), va='center', ha='center', fontsize = 24)

    # Draw a vertical line down the middle
    plt.axvline(base, color='black')

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    axes = plt.gca()  # (gca = get current axes)
    axes.spines['left'].set_visible(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.xaxis.set_ticks_position('bottom')

    # Make the y-axis display the variables
    plt.yticks(ys, variables, fontsize = 18)

    # Set the portion of the x- and y-axes to show
    #plt.xlim(base - 1000, base + 1000)
    plt.ylim(-1, len(variables))
    plt.xlabel(r'$S_p^q$', fontsize = 18)
    #plt.title(title_string, fontsize = 18)
    plt.show()

plot_tornado(state_var='V_ss', col = 'blue')
plot_tornado(state_var = 'I_ss', col = 'red')
plot_tornado(state_var = 'Re', col = 'black')
