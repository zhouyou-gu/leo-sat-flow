import os
from os.path import expanduser

from os.path import dirname, abspath

def get_working_dir_path():
    return dirname(abspath(__file__))
