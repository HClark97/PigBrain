import scipy.io
import plyer as pl
import ctypes    
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
        Contains noxious data
    
    Help
    -------
    Data is accesed using the following syntax:
    data['Non/nox']['block'][i]['channel'][i]['ERPs'][:,i]
    
    """
    
    if str(pl.utils.platform) == 'macosx':
        path=list(['/Users/amaliekoch/Dropbox (Personlig)/Aalborg universitet/8. semester/Projekt/ERPs_Subject1'])
    else:
        path=pl.filechooser.open_file(filters=[("Binary MATLAB file (*.mat)", "*.mat")])
    
    if len(path):
            data = scipy.io.loadmat(path[0],simplify_cells=True)
            data.pop('__globals__')
            data.pop('__header__')
            data.pop('__version__')
            return data
            
    elif str(pl.utils.platform) == 'win':
            ctypes.windll.user32.MessageBoxW(0, "Please select correct file path", "Path not found")
    
        



