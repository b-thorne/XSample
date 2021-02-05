import os

from absl import app
from absl import flags
from absl import logging

import gin

import numpy as np
import classy

import matplotlib.pyplot as plt
import cosmoplotian.colormaps
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

string_cmap = "div yel grn"
#string_cmap = "RdYlBu"
cmap = mpl.cm.get_cmap(string_cmap)
plt.rcParams['text.usetex'] = True

from tqdm import tqdm
from pathlib import Path
import pickle
import copy
FLAGS = flags.FLAGS

NUM_CONSTANT = 40
NUM_ANCHORS = 10

LMAX = 2000

DEFAULT_PARAMS = {
    'output': 'tCl lCl',
    'l_max_scalars': LMAX, 
    'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.9624,
    #'h': 0.6711,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029,
    #'100*theta_s': 1.05256443,
    'dsg_log10a_vals': "-15.,-14.673,-14.347,-14.02,-13.694,-13.367,-13.041,-12.714,-12.388,-12.061,-11.735,-11.408,-11.082,-10.755,-10.429,-10.102,-9.7755,-9.449,-9.1224,-8.7959,-8.4694,-8.1429,-7.8163,-7.4898,-7.1633,-6.8367,-6.5102,-6.1837,-5.8571,-5.5306,-5.2041,-4.8776,-4.551,-4.2245,-3.898,-3.5714,-3.2449,-2.9184,-2.5918,-2.2653,-1.9388,-1.6122,-1.2857,-0.95918,-0.63265,-0.30612,0.020408,0.34694,0.67347,1.",
    'dsg_w_vals': "0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333,0.333333333333333",
    'dsg_c_eff2': 0.33333333,
    'nap': 'y',
    'dsg_c_vis2': 0.33333333,
    'dsg_alpha': 0.00001200901405,
}


def GetDefaultParams():
    return copy.deepcopy(DEFAULT_PARAMS)


def GetSpectra(params, type='lensed', lmax=2000):
    cosmo = classy.Class()
    cosmo.set(params)
    cosmo.compute()
    computed_cls = cosmo.lensed_cl(lmax)
    c2d = computed_cls['ell'] * (computed_cls['ell'] + 1) / 2. / np.pi * (2.725e6 ** 2) 
    cosmo.struct_cleanup()
    cosmo.empty()
    return computed_cls, c2d


def insert_inset_colorbar(fig, ax, vmin, vmax, label=None, orientation="horizontal", loc="upper center", width="30%", height="3%"):
    axins = inset_axes(ax, width=width, height=height, loc=loc)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=axins, orientation=orientation, label=label)
    return

def constant_wg(w):
    return ",".join([f"{0.3333333:.05f}" for _ in range(NUM_CONSTANT)] + [f"{w:.05f}" for _ in range(NUM_ANCHORS)])

def ExploreDesignerHz(results_dir, N=5):
    """ Function runs CLASS varying one parameter at a time over a set range. Plots each
    of these parameter variations in a subplot.
    """
    params = GetDefaultParams()

    vmin = -1/6.
    vmax = 0.3
    w_range = np.linspace(vmin, vmax, N)
    
    # Make plot varying w weights uniforml (with some early weight set to 1/3)
    logging.info("Designer: Varing w weights uniformly ...")
    fig, ax = plt.subplots(1, 1)
    for i, w in enumerate(tqdm(w_range, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}')):
        params = GetDefaultParams()
        params['dsg_w_vals'] = constant_wg(w)

        computed_cls, c2d = GetSpectra(params, type='lensed', lmax=LMAX)
        insert_inset_colorbar(fig, ax, vmin, vmax, r"$w$")
        ax.plot(computed_cls['ell'], np.sqrt(c2d * computed_cls['tt']), color=cmap(i / N))
    
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.set_xlim(2, LMAX)
    ax.set_yscale('log')
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\left[\ell(\ell+1)/2\pi\right] C_\ell~({\rm \mu K})$")
 
    fig.savefig(results_dir / "DesignerHz_spectra_w.pdf", bbox_inches="tight")

    # Make plot varying c_eff2
    logging.info("Designer: Varying c_eff2 ... ")
    fig, ax = plt.subplots(1, 1)
    vmin = 0
    vmax = 1
    dsg_c_eff2 = np.linspace(vmin, vmax, N)
    for i, c in enumerate(tqdm(dsg_c_eff2, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}')):
        params = GetDefaultParams()
        params['dsg_w_vals'] = ",".join([f"{0.01:.05f}" for _ in range(NUM_CONSTANT+NUM_ANCHORS)])
        params['dsg_alpha'] = 0.001
        params['dsg_c_eff2'] = c
        computed_cls, c2d = GetSpectra(params, type='lensed', lmax=LMAX)
        insert_inset_colorbar(fig, ax, vmin, vmax, r"$c_{\rm eff}^2$")
        ax.plot(computed_cls['ell'], np.sqrt(c2d * computed_cls['tt']), color=cmap(i / N))
    
    ax.tick_params(axis="both", direction="in", right=True, top=True)
    ax.set_xlim(2, 2000)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\left[\ell(\ell+1)/2\pi C_\ell\right]^{1/2}~({\rm \mu K})$")

    fig.savefig(results_dir / "DesignerHz_spectra_ceff2.pdf", bbox_inches="tight")

    return



def ExploreLCDM(results_dir, N=5):  
    """ Function runs CLASS varying one parameter at a time over a set range. Plots each
    of these parameter variations in a subplot.
    """
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    ranges = {
        'omega_b': {
            'lims': [0.01, 0.03],
            'label': r"$\Omega_b$"
        },
        'omega_cdm': {
            'lims': [0.1, 0.5], 
            'label': r"$\Omega_c$",
        },
        'n_s': {
            'lims': [0.94, 1.05],
            'label': r"$n_s$"
        },
        'A_s': {
            'lims': [2.3e-9, 3.3e-9],
            'label': r"$10^9 \log A_s$"
        }
    }

    logging.info("LCDM: Varying basic parameters ... ")
    for ax_i, (k, settings) in enumerate(tqdm(ranges.items(), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}')):
        params = GetDefaultParams()
        vmin, vmax = settings['lims']
        param_range = np.linspace(vmin, vmax, N)
        ax = axes.flatten()[ax_i]
        for i in range(N):
            params[k] = param_range[i]
            computed_cls, c2d = GetSpectra(params, 'lensed')
            if k == 'A_s':
                vmin *= 1e9
                vmax *= 1e9
            insert_inset_colorbar(fig, ax, vmin, vmax, settings['label'])
            ax.plot(computed_cls['ell'], np.sqrt(c2d * computed_cls['tt']), color=cmap(i / N))

    for ax in axes.flatten():
        ax.set_yscale('linear')
        ax.set_ylim(0, 110)
        ax.set_xlim(2, 2000)

    axes[0, 0].set_ylabel(r"$\left[\ell(\ell + 1)/2\pi C_\ell\right]^{1/2}~({\rm \mu K})$")
    axes[1, 0].set_ylabel(r"$\left[\ell(\ell + 1)/2\pi C_\ell\right]^{1/2}~({\rm \mu K})$")
    axes[1, 1].set_xlabel(r"$\ell$")
    axes[1, 0].set_xlabel(r"$\ell$")

    fig.savefig(results_dir / "LCDM_spectra.pdf")

    return

def main(argv):
    del argv # unused

    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if FLAGS.mode == "All":
        ExploreLCDM(results_dir)
        ExploreDesignerHz(results_dir)
    if FLAGS.mode == "LCDM":
        ExploreLCDM(results_dir)
    if FLAGS.mode == "DesignerHz":
        ExploreDesignerHz(results_dir)

    return

if __name__ == "__main__":
    flags.DEFINE_enum(
        "mode", 
        "DesignerHz", 
        ["All", "DesignerHz", "LCDM"], 
        "Which mode to run in.")
    flags.DEFINE_string("results_dir", "./results/spectra", "Directory in which to save resulting plots.")
    app.run(main)