import scipy.io
import plyer as pl

def load_mat_file():
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
    path=pl.filechooser.open_file()
    f = scipy.io.loadmat(path[0],simplify_cells=True)
    Non=f['NonnoxERP']
    Nox=f['NoxERP']
    
    return Non,Nox
    





