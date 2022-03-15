import scipy.io
import h5py as h5py
import numpy as np

def load_mat_file(path):
    """
    Parameters
    ----------
    path : string
        DESCRIPTION.

    Returns
    -------
    Non : dict
        Contains non-noxious data
    Nox : dict
        Contains non-noxious data
    
    Help
    -------
    Data is accesed using the following syntax:
    dict['block'][i]['channel'][i]['ERPs'][i]
    
    """
    f = scipy.io.loadmat(path,simplify_cells=True)
    f.pop('__globals__')
    f.pop('__header__')
    f.pop('__version__')
    data = np.array(list(f.items()))
    Non=data[0,1]
    Nox=data[1,1]
    
    return Non,Nox
    





