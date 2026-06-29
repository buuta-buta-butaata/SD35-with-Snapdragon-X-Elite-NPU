import os
import numpy as np
from datetime import datetime

def save(dir_path, file_name, data):
    # file_name = file_name + datetime.now().strftime("%Y%m%d%H%M%S") + ".npy"
    save_path = os.path.join(dir_path, file_name)
    np.save(save_path, data)

def load(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data
