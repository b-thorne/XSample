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

import requests

FLAGS = flags.FLAGS

DATASET_DIRECTORY = "/global/cscratch1/sd/bthorne/XSample/datasets"

BASELINE_CHAINS_URL = "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip"
BAELINE_CHAINS_DIR = "/global/cscratch1/sd/bthorne/XSample/chains/baseline_LCDM/base/plikHM_TTTEEE_lowl_lowE"

GNILC_FNAME = "COM_CompMap_Dust-GNILC-F545_2048_R2.00.fits"


STRING_TO_CHAIN = {
    "base_plikHM_TTTEEE_lowl_lowE": Path(BAELINE_CHAINS_DIR) / "base_plikHM_TTTEEE_lowl_lowE.npy"
}

def prepare_base_LCDM_plikHM_TTTEEE_lowl_lowE():
    chains = [np.genfromtxt(Path(BAELINE_CHAINS_DIR) / f"base_plikHM_TTTEEE_lowl_lowE_{i+1}.txt") for i in trange(4)]
    chains = np.concatenate(chains)
    logging.info("Saving to " + str(STRING_TO_CHAIN["base_plikHM_TTTEEE_lowl_lowE"]))
    np.save(STRING_TO_CHAIN["base_plikHM_TTTEEE_lowl_lowE"], chains)
    return


def load_dataset(label):
    samples = np.load(STRING_TO_CHAIN[label])
    paramnames = np.genfromtxt(str(Path(BAELINE_CHAINS_DIR) / label) + ".paramnames", delimiter="\t", dtype=('U10', 'U10'))
    dataset_info = {
        'nsamples': samples.shape[0],
        'nparams': samples.shape[-1],
        'paramnames': paramnames[:, 0],
        'paramnames_latex': paramnames[:, 1]
    }
    return samples, dataset_info


def main(argv):
    del argv

    if FLAGS.dataset == "base_LCDM_plikHM_TTTEEE_lowl_lowE":
        prepare_base_LCDM_plikHM_TTTEEE_lowl_lowE()
    
    return


if __name__ == '__main__':
    flags.DEFINE_enum(
    "dataset", 
    "base_LCDM_plikHM_TTTEEE_lowl_lowE", 
    ["base_LCDM_plikHM_TTTEEE_lowl_lowE"], 
    "Which datset to prepare.")
    app.run(main)