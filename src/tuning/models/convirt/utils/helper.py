"""
Helper functions.
"""
import json
import os
import logging
import pydicom
import random
import numpy as np
import torch

logger = logging.getLogger('transfer')

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Utils for data loading

def get_image_from_dicom(dicom_file):
    """
    Extract the image as an array from a DICOM file.
    """
    dcm = pydicom.read_file(dicom_file)
    array = dcm.pixel_array
    try:
        array *= int(dcm.RescaleSlope)
        array += int(dcm.RescaleIntercept)
    except:
        pass
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        array = np.invert(array.astype("uint16"))
    array = array.astype("float32")
    array -= np.min(array)
    array /= np.max(array)
    array *= 255.
    return array.astype('uint8')

# Utils for IO
def check_dir(d):
    if not os.path.exists(d):
        logger.error("Directory {} does not exist. Exit.".format(d))
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            logger.error("File {} does not exist. Exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            logger.info("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        logger.info("Config saved to file {}".format(path))
    return config

def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        logger.info("Config loaded from file {}".format(path))
    return config

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    logger.info(info + "\n")
    return
