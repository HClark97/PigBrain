# -*- coding: utf-8 -*-
import h5py
import numpy as np
import json
from sys import platform

def get_base_path():
    filepath = '.config'
    # Try opening the path file
    try:
        with open(filepath, 'r') as pathfile:
            jsonData = json.load(pathfile)
            path = jsonData['path']
            if path:
                return path
            else:
                print("Failed to read path from config file...")
                return None
    except IOError:
        try:
            with open(filepath, 'w') as pathfile:
                path = input("Please input the basepath to the data directory: ")
                jsonData = {
                    'path': path
                }
                pathfile.write(json.dumps(jsonData))
                return path
        except IOError:
            print("Failed to save path to config file...")
            return None
    except:
        print("Failed to handle config file with unknown error...")
        return None

def get_os():
	# Run install function for corrospondig OS
	if platform == 'win32' or platform == "windows":
		return 0
	if platform.startswith('linux'):
		return 1
	if platform == 'darwin': # MacOS
		return 2

def get_file_path(group, exp, block, bset, filename):
    # Construct the filepath
    if(get_os() == 0):
        return r'{}'.format('{}\\{}\\Exp {}\\Block {}\\Set {}\\{}'.format(get_base_path(), group, exp, block, bset, filename).replace('"',''))
    elif (get_os() == 1 or get_os() == 2):
        return r'{}'.format('{}/{}/Exp {}/Block {}/Set {}/{}'.format(get_base_path(), group, exp, block, bset, filename).replace('"',''))
    else:
        raise Exception("Unable to get filepath")

def get_relative_filepath(subfolders, filename):
    path = "."
    if get_os() == 0:
        for folder in subfolders:
            path = '{}\\{}'.format(path, folder)
        path = '{}\\{}'.format(path, filename)
    elif get_os() == 1 or get_os() == 2:
        for folder in subfolders:
            path = '{}/{}'.format(path, folder)
        path = '{}/{}'.format(path, filename)

def save_mat_file(path, data, fs):
    save_file = h5py.File(path,"w")
    file_group = save_file.create_group('Array_storage')
    file_group.create_dataset('array_format_data', data=data)
    file_group.create_dataset('fs',(1,1), data=np.array(fs))
    # Close the file
    save_file.close()
    
def load_mat_file(path):
    file = h5py.File(path, "r")
    fs = file['Array_storage']['fs'][0,0]
    data = file['Array_storage']['array_format_data'][()]
    return data, fs
