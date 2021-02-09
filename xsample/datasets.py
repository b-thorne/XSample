# Datasets
# --------
#
# This file is responsible for reading in datasets saved in `CHAINS_DIR`.
#
# We also make some plots of different dataset properties.  
# - Compare to top row in Figure 26 of 1807.06209.
# - Compare to Figure 29 of 1807.06209.
import os

from absl import flags
from absl import app
from absl import logging

import numpy as np

from pathlib import Path 

import matplotlib as mpl
import matplotlib.pyplot as plt 
import cosmoplotian
import cosmoplotian.colormaps
cmap = mpl.cm.get_cmap("div yel grn")
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", "red"]) 
plt.rcParams['text.usetex'] = True

import getdist
from getdist import plots

FLAGS = flags.FLAGS

# warning this is an 11 GB file.
CHAINS_DIR = "/global/cscratch1/sd/bthorne/XSample"

STRING_TO_CHAIN = {
    "base_plikHM_TTTEEE_lowl_lowE": Path(CHAINS_DIR) / "base" / "plikHM_TTTEEE_lowl_lowE" / "base_plikHM_TTTEEE_lowl_lowE",
    "base_omegak_plikHM_TTTEEE_lowl_lowE": Path(CHAINS_DIR) / "base_omegak"  / "plikHM_TTTEEE_lowl_lowE" /  "base_omegak_plikHM_TTTEEE_lowl_lowE",
    "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO": Path(CHAINS_DIR) / "base_omegak"  / "plikHM_TTTEEE_lowl_lowE_BAO" /  "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO",
}

ALAM_2017_BAO_DIR =  Path(CHAINS_DIR) / "ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints" / "COMBINEDDR12_BAO_consensus_dM_Hz"
RD_FID = 147.78 # in Mpc


def LoadDataset(label):
    return getdist.loadMCSamples(str(STRING_TO_CHAIN[label])) 

def LoadBAODataset():
    # Grabbed most of this function from MontePython
    data_file = ALAM_2017_BAO_DIR / "BAO_consensus_results_dM_Hz.txt"
    cov_file = ALAM_2017_BAO_DIR / "BAO_consensus_covtot_dM_Hz.txt"
    z = np.array([], 'float64')
    DM_rdfid_by_rd_in_Mpc = np.array([], 'float64')
    H_rd_by_rdfid_in_km_per_s_per_Mpc = np.array([], 'float64')
    # read redshifts and data points
    with open(data_file, 'r') as filein:
        for i, line in enumerate(filein):
            if line.strip() and line.find('#') == -1:
                this_line = line.split()
                # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                if this_line[1] == 'dM(rsfid/rs)':
                    z = np.append(z, float(this_line[0]))
                    DM_rdfid_by_rd_in_Mpc = np.append(
                        DM_rdfid_by_rd_in_Mpc, float(this_line[2]))
                # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                elif this_line[1] == 'Hz(rs/rsfid)':
                    H_rd_by_rdfid_in_km_per_s_per_Mpc = np.append(
                        H_rd_by_rdfid_in_km_per_s_per_Mpc, float(this_line[2]))
    cov_data = np.loadtxt(cov_file)
    
    return z, DM_rdfid_by_rd_in_Mpc, H_rd_by_rdfid_in_km_per_s_per_Mpc, cov_data, RD_FID

def Plot1D_Omegak_H0_Omegam(results_dir):
    labels = [
        "base_omegak_plikHM_TTTEEE_lowl_lowE",
        "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO"
    ]
    datasets = [LoadDataset(label) for label in labels]
    
    g = plots.get_subplot_plotter(width_inch=5)
    g.settings.figure_legend_frame = False
    g.plots_1d(datasets, ['omegak', 'H0', 'omegam'], nx=3, legend_labels=["PlikHM TTTEEE+lowl+lowE", "+BAO"], legend_ncol=2)
    g.export(str(results_dir / "1D_omegak_H0_omegam.pdf"))
    return 


def Plot2D_Omegak(results_dir):
    labels = [
        "base_omegak_plikHM_TTTEEE_lowl_lowE",
        "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO"
    ]
    datasets = [LoadDataset(label) for label in labels]
    
    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_frame = False
    g.plot_2d(datasets, ['omegak', 'omegam'], filled=True)
    g.add_legend(["PlikHM TTTEEE+lowl+lowE", "+BAO"], frameon=False)
    g.export(str(results_dir / "2D_omegak_omegam.pdf"))
    return 


def Plot3D_Omegak_Omegam_H0(results_dir):
    labels = [
        "base_omegak_plikHM_TTTEEE_lowl_lowE",
        "base_omegak_plikHM_TTTEEE_lowl_lowE_BAO"
    ]
    datasets = [LoadDataset(label) for label in labels]

    g = plots.get_single_plotter(width_inch=5)
    g.settings.legend_frame = False
    g.plots_3d(datasets, [['omegak', 'omegam', 'H0'], ['omegak', 'omegam', 'H0']], nx=2, legend_labels=["PlikHM TTTEEE+lowl+lowE", "+BAO"], legend_ncol=2)
    #g.add_legend(["PlikHM TTTEEE+lowl+lowE", "+BAO"])
    g.export(str(results_dir / '3D_omega_k_omegam_h0.pdf'))
    return 


def main(argv):
    del argv
    # would be nice to put some plots of default datasets here
    results_dir = Path(FLAGS.results_dir) 
    Plot1D_Omegak_H0_Omegam(results_dir)
    Plot2D_Omegak(results_dir)
    Plot3D_Omegak_Omegam_H0(results_dir)
    return


if __name__ == '__main__':
    flags.DEFINE_string(
        "results_dir", 
        "results/datasets",
        "Directory to save plots.")
    app.run(main)