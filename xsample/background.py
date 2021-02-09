from absl import flags
from absl import app

import numpy as np 
from scipy.integrate import quad
from dataclasses import dataclass, field

from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib as mpl
import cosmoplotian.colormaps

from spectra import insert_inset_colorbar
from datasets import LoadBAODataset

string_cmap = "div yel grn"
#string_cmap = "RdYlBu"
cmap = mpl.cm.get_cmap(string_cmap)
plt.rcParams['text.usetex'] = True
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", cmap(1)]) 

FLAGS = flags.FLAGS

km = 6.187152231383511e37
second = 1.8548615754666474e43
Kelvin = 7.058235349009319e-33
eV = 8.190723190477751e-29
Mpc = 1.90916956403801e57
gram = 45946.59168182904
Joule = 5.11221307704105e-10
me = 510.9989461 * eV

H0units = km/second/Mpc
rhox_over_omegax = 3 * (100 * H0units) ** 2 / (8 * np.pi)

@dataclass
class Params:
    omega_b: field(default=0.0225)
    omega_c: field(default=0.12)
    H0: field(default=67.0)
    Nnu_massive: field(default=1.0)
    Nnu_massless: field(default=2.046)
    mnu: field(default=0.06)
    Omega_k: field(default=0.)
    Tcmb: field(default=2.7255)
    w: field(default=-1)

    def __post_init__(self):
        self.hsquared = (self.H0 / 100.) ** 2
        self.T_gamma_0 = self.Tcmb * Kelvin
        self.rho_gamma_0 = (np.pi ** 2 / 15.) * self.T_gamma_0 ** 4
        self.omega_gamma = self.rho_gamma_0 / rhox_over_omegax
        self.Omega_gamma = self.omega_gamma / self.hsquared
        self.rho_b_0 = self.omega_b * rhox_over_omegax
        self.Omega_b = self.omega_b / self.hsquared
        self.rho_c_0 = self.omega_c * rhox_over_omegax
        self.Omega_c = self.omega_c / self.hsquared
        if self.mnu == 0:
            self.Nnu_massless += self.Nnu_massive
            self.Nnu_massive = 0
        self.rho_nu_0 = rho_nu(0, self)
        self.omega_nu = self.rho_nu_0 / rhox_over_omegax
        self.Omega_nu = self.omega_nu / self.hsquared
        self.omega_k = self.Omega_k * self.hsquared
        self.rho_k_0 = self.omega_k * rhox_over_omegax
        self.Omega_lambda = 1 - self.Omega_k - self.Omega_b - self.Omega_c - self.Omega_gamma
        self.omega_lambda = self.Omega_lambda * self.hsquared
        self.rho_lambda_0 = self.omega_lambda * rhox_over_omegax
        return

def rho_gamma(z, p):
    return p.rho_gamma_0 * (1 + z) ** 4

def rho_lambda(z, p):
    return p.rho_lambda_0 * (1 + z) ** (3 * (1 + p.w))

def rho_c(z, p):
    return p.rho_c_0 * (1 + z) ** 3

def rho_b(z, p):
    return p.rho_b_0 * (1 + z) ** 3
    
def rho_nu(z, p):
    return rho_nu_massless(z, p) + rho_nu_massive(z, p)

def rho_nu_massless(z, p):
    return p.Nnu_massless * (7. / 8. * (4. / 11.) ** (4. / 3.)) * p.rho_gamma_0 * (1 + z) ** 4

def rho_nu_massive(z, p):
    if p.mnu == 0:
        return 0
    else:
        a = 1 / (1. + z)
        a_prime = 1e-7
        Tnu = p.T_gamma_0 * (4. / 11.) ** (1. / 3.)
        mTsquared = (p.mnu / (Tnu / a_prime)) ** 2
        def integrand(x):
            return x ** 2 * np.sqrt(((x * a_prime / a)) ** 2 + mTsquared) / (np.exp(np.sqrt(x ** 2 + mTsquared)) + 1)   
        return p.Nnu_massive * 1 / np.pi ** 2 * (Tnu / a_prime) ** 4 * (a_prime / a) ** 3 * quad(integrand, 0, np.inf)[0]

def conformal_time_between(z1, z2, p):
    return quad(lambda z: 1 / H(z, p), z1, z2)[0]

def conformal_time(z, p):
    return conformal_time_between(z, 0, p)

def comoving_distance(z, p):
    return - conformal_time(z, p)

def angular_diameter_distance(z, p):
    cd = comoving_distance(z, p)
    K = - p.omega_k * (100 * km / second / Mpc) ** 2
    if K == 0:
        return cd 
    elif K < 0:
        return 1 / np.sqrt(-K) * np.sin(cd * np.sqrt(- K))
    elif K > 0:
        return 1 / np.sqrt(K) * np.sinh(cd * np.sqrt(K))
    
def H(z, p):
    return np.sqrt(8 * np.pi / 3.) * np.sqrt(p.rho_k_0 * (1 + z) ** 2 + rho_b(z, p) + rho_c(z, p) + rho_lambda(z, p) + rho_gamma(z, p) + rho_nu(z, p))

def PlotBAOData(results_dir):
    vfunc = np.vectorize(angular_diameter_distance)
    redshifts = np.logspace(-2, np.log10(2))
    
    z, DArd, Hzrd, cov, rd_fid = LoadBAODataset()
    fig, ax = plt.subplots(1, 1)
    vmin = -0.05
    vmax = 0.05
    N = 10
    for i, Omega_k in enumerate(np.linspace(vmin, vmax, N)):
        pars = Params(omega_b=0.0225,omega_c=0.12,H0=67.0,Nnu_massive=1.0,Nnu_massless=2.046,mnu=0.06, Omega_k=Omega_k,Tcmb=2.7255, w=-1) 
        ax.loglog(redshifts, vfunc(redshifts, pars) / Mpc, label=f"{Omega_k}", color=cmap(i / N))
    ax.errorbar(z, DArd, fmt='o', yerr=np.sqrt(np.diag(cov)[[0, 2, 4]]), color='k')
    ax.set_yscale('linear')
    ax.set_xlim(0.01, 2)
    ax.set_ylim(0, 5500)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$D_A(z)~({\rm Mpc})$")
    insert_inset_colorbar(fig, ax, vmin, vmax, r"$\Omega_k$")
    fig.savefig(results_dir / "DR12_BAO_data.pdf")
    return

def main(argv):
    del argv 

    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if FLAGS.mode == "PlotBAOData":
        PlotBAOData(results_dir)

if __name__ == "__main__":
    flags.DEFINE_enum(
        "mode", 
        "PlotBAOData", 
        ["PlotBAOData"], 
        "Which mode to run in.")
    flags.DEFINE_string(
        "results_dir", 
        "./results/background", 
        "Directory to write results.")
    app.run(main)