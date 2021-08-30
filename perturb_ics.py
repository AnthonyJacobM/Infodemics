import numpy as np
from scipy.integrate import odeint
import pandas as pd

def perturb_ics(ics_dict, eps0):
    """
    funciton to perturb any initial conditions
    :param ics_dict: dictionary of initial conditions
    :param eps0: weight of noise
    :return: perturbed initial conditions
    """
    ss_dict_perturbed = {}

    # generate new initial conditions!
    for k, v in ics_dict.items():
        sgn = np.sign(np.random.rand(1) - 1 / 2)[0]
        ss_dict_perturbed[k] = np.round(v, 6) * (1 + sgn * eps0 * np.random.rand(1))[0]  # perturb new steady state

    return ss_dict_perturbed


