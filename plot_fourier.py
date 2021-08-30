import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from gen_sys import gen_sys
from scipy.fftpack import fft, ifft, fftn, ifftn
from perturb_ics import perturb_ics

# initialize parameter and intial conditions for the numerical simulation
par_dict_def = {'recovery': 0.07, 'belief': 1.0,
            'risk': 0.10, 'protection': 0.90,
            'education': 0.33, 'misinformation': 0.10,
            'infection_good': 0.048, 'infection_bad': 0.37, 'ace': 0}

# x1 ~ sg, x2 ~ sb, x3 ~ ib, x4 ~ v, x5 ~ phi
ics_dict_def = {'x1': 0.30, 'x2': 0.55,
            'x3': 0.01, 'x4': 0.0,
            'x5': 0.50}

eq1_h1_ss = {'x1': 0.1652553343953094,
             'x2': 0.4608116686366218,
             'x3': 0.09068387295130048,
             'x4': 0.14189412748039304,
             'x5': 0.0003737491655869812}

eq1_h1_par_dict = par_dict_def
eq1_h1_par_dict['risk'] = 1.635295791362042

def fourier_fft(t, y, show_bool=False, m0 = None):
    """
    function to generate a fourier transform of the data
    :param t: time series
    :param y: y-argument of the fourier domain
    :param show_bool: boolean to show the plot
    :param m0: size of the start of the periodic sequence
    :return:
    """
    if m0 == None:
        m = int(len(t) / 2)
    else:
        m = m0

    # change the size of the data
    y = y[m:]
    t = t[m:]

    # now we determine a function for using the fourier transform
    a = t[0]
    b = t[-1]
    T = b - a
    N = len(t)
    delta_t = T / N
    delta_f = 1 / T
    n_limit = int(N / 2)
    f0 = 0

    # generate the fourier transform
    yf = fft(y[0:n_limit])
    # normalize
    yf = yf / len(yf)
    power = np.abs(yf)
    phase = np.angle(yf)

    # convert it to the data
    phase = phase * 180 / np.pi

    ynew = y[0:int(n_limit)]
    fmax = delta_f * n_limit
    freq = np.linspace(f0, fmax, len(ynew))

    # now we plot provided the boolean is true
    if show_bool == True:
        plt.plot(t, y, 'b', lw=2)
        plt.xlabel('t', fontsize=18)
        plt.ylabel('y', fontsize=18)
        plt.show()

        plt.plot(freq, power, 'r', lw=2)
        plt.xlabel(r'$days^{-1}$', fontsize=18)
        plt.ylabel('Amplitude', fontsize=18)
        plt.show()

        plt.plot(freq, phase, 'k', lw=2)
        plt.xlabel(r'$days^{-1}$', fontsize=18)
        plt.ylabel('Phase', fontsize=18)
        plt.show()

    return freq, power, phase, yf



def sample():
    """
    function to plot a sampled fourier space
    :return: plotted solution
    """
    # generate a sys
    ss_perturb = perturb_ics(ics_dict=eq1_h1_ss, eps0 = 0.01)

    # genrate an ode systme
    z, t = gen_sys(par_dict = eq1_h1_par_dict, ics_dict=ss_perturb, tf = 10_000)

    # use the infected to plot the fourier
    sg, sb, ib, v, phi = z.T
    m = int(len(t) / 2)
    sg = sg[m:]
    sb = sb[m:]
    ib = ib[m:]
    v = v[m:]
    phi = phi[m:]
    t = t[m:]

    # generate the distribution
    freq_inf, power_inf, phase_inf, yf_inf = fourier_fft(t, 1 - (sg + sb + v), show_bool=True)
    freq_bad, power_bad, phase_bad, yf_bad = fourier_fft(t, sb + ib, show_bool=True)
    freq_delta_e, power_delta_e, phase_delta_e, yf_delta_e = fourier_fft(t, 1 - (sg + sb + v) - eq1_h1_par_dict['risk'] * v, show_bool=True)



#sample()