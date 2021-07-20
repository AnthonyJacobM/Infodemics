from PyDSTool import *
import PyDSTool as dst

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