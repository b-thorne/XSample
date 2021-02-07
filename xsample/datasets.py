import tensorflow as tf 

from absl import flags
from absl import app
from absl import logging

import numpy as np

from tqdm import tqdm
from tqdm import trange
from pathlib import Path
from contextlib import contextmanager 
import os
import sys
from collections import namedtuple

import getdist

import requests

FLAGS = flags.FLAGS

# warning this is an 11 GB file.
BASELINE_CHAINS_URL = "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip"
CHAINS_DIR = "/global/cscratch1/sd/bthorne/XSample"

STRING_TO_CHAIN = {
    "base_plikHM_TTTEEE_lowl_lowE": Path(CHAINS_DIR) / "base" / "plikHM_TTTEEE_lowl_lowE" / "base_plikHM_TTTEEE_lowl_lowE",
    "base_omegak_plikHM_TTTEEE_lowl_lowE": Path(CHAINS_DIR) / "base_omegak"  / "plikHM_TTTEEE_lowl_lowE" /  "base_omegak_plikHM_TTTEEE_lowl_lowE",
    "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO": Path(CHAINS_DIR) / "base_omegak"  / "plikHM_TTTEEE_lowl_lowE_BAO" /  "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO",
}

def load_dataset(label):
    return getdist.loadMCSamples(str(STRING_TO_CHAIN[label])) 

def main(argv):
    del argv
    # would be nice to put some plots of default datasets here
    return


if __name__ == '__main__':
    flags.DEFINE_enum(
    "dataset", 
    "base_LCDM_plikHM_TTTEEE_lowl_lowE", 
    ["base_LCDM_plikHM_TTTEEE_lowl_lowE"], 
    "Which datset to prepare.")
    app.run(main)