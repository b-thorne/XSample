from absl import flags
from absl import app

import numpy as np 
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from dataclasses import dataclass, field

from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib as mpl
import cosmoplotian.colormaps

from spectra import insert_inset_colorbar
from datasets import LoadBAODataset
from recfast4py import recfast

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
sigma_T = 2.546617863799246e41
mH = 7.690195407123562e-20

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
    Gamma_P: field(default=0.24)
    F: field(default=1.14) # RECFAST parameter
    fDM: field(default=0.) # RECFAST parameter

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
    
def drs_dz(z, p):
    R = 3 * p.rho_b_0 / (4 * p.rho_gamma_0 * (1 + z))
    return 1 / H(z, p) / np.sqrt(3 * (1 + R))

def sound_horizon(z, p):
    def integrand(x):
        return drs_dz(x, p)
    return quad(integrand, z, np.inf)[0]

def r_drag(p):
    return sound_horizon(z_drag(p), p)

def z_drag(p):
    def root_func(x):
        return tau_drag(0, x, p) -1
    roots = root_scalar(root_func, x0=800, x1=1400)
    return roots.root

def tau_drag(z1, z2, p):
    R_div_a = 3 * p.rho_b_0 / (4. * p.rho_gamma_0)
    zarr, Xe_H, Xe_He, Xe, TM = recfast.Xe_frac(p.Gamma_P, p.Tcmb, p.Omega_c, p.Omega_b, p.Omega_lambda, p.Omega_k, np.sqrt(p.hsquared), p.Nnu_massless, p.F, p.fDM)
    xe = interp1d(zarr, Xe, kind='cubic')
    def integrand(x):
        return xe(x) / H(x, p) * (1 + x) ** 3
    return  sigma_T * p.rho_b_0 / mH * (1 - p.Gamma_P) / R_div_a * quad(integrand, z1, z2)[0]

def H(z, p):
    return np.sqrt(8 * np.pi / 3.) * np.sqrt(p.rho_k_0 * (1 + z) ** 2 + rho_b(z, p) + rho_c(z, p) + rho_lambda(z, p) + rho_gamma(z, p) + rho_nu(z, p))

def PlotBAOData(results_dir):
    vfunc = np.vectorize(angular_diameter_distance)
    redshifts = np.logspace(-2, np.log10(2))
    
    z, DArd, Hzrd, cov, rd_fid = LoadBAODataset()
    fig, ax = plt.subplots(1, 1)
    vmin = -0.1
    vmax = 0.1
    N = 10
    for i, Omega_k in enumerate(np.linspace(vmin, vmax, N)):
        pars = Params(omega_b=0.0225,omega_c=0.12,H0=67.0,Nnu_massive=1.0,Nnu_massless=2.046,mnu=0.06, Omega_k=Omega_k,Tcmb=2.7255, w=-1, Gamma_P=0.24, F=1.14, fDM=0.) 
        ax.loglog(redshifts, vfunc(redshifts, pars) / Mpc, color=cmap(i / N), alpha=0.5)
    ax.errorbar(z, DArd, fmt='o', yerr=np.sqrt(np.diag(cov)[[0, 2, 4]]), color='k', label="BOSS DR12")
    pars = Params(omega_b=0.0225,omega_c=0.12,H0=67.0,Nnu_massive=1.0,Nnu_massless=2.046,mnu=0.06, Omega_k=0.,Tcmb=2.7255, w=-1, Gamma_P=0.24, F=1.14, fDM=0.) 
    ax.loglog(redshifts, vfunc(redshifts, pars) / Mpc, color='k', label=r"$\Lambda {\rm CDM}$")
    ax.set_yscale('linear')
    ax.set_xlim(0.01, 2)
    ax.set_ylim(0, 5500)
    ax.set_xlabel(r"${\rm Redshift,}~z$")
    ax.set_ylabel(r"${\rm Angular~Diameter~Distance,}~D_A(z)~({\rm Mpc})$")
    ax.tick_params(direction="inout", axis="both")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc=2, frameon=False)
    insert_inset_colorbar(fig, ax, vmin, vmax, r"$\Omega_k$")
    fig.savefig(results_dir / "DR12_BAO_data.pdf")
    return

def PlotRECFAST(results_dir):
    pars = Params(omega_b=0.0225,omega_c=0.12,H0=67.0,Nnu_massive=1.0,Nnu_massless=2.046,mnu=0.06, Omega_k=0.,Tcmb=2.7255, w=-1, Gamma_P=0.24, F=1.14, fDM=0.)
        
    zarr, Xe_H, Xe_He, Xe ,TM = recfast.Xe_frac(pars.Gamma_P, pars.Tcmb, pars.Omega_c, pars.Omega_b, pars.Omega_lambda, pars.Omega_k, np.sqrt(pars.hsquared), pars.Nnu_massless, pars.F, pars.fDM)
    fig, ax = plt.subplots(1, 1)
    ax.plot(zarr, Xe, 'k-', label=r"$X_{\rm e}$")
    ax.plot(zarr, Xe_H, 'k--', label=r"$X_{\rm e, H}$")
    ax.plot(zarr, Xe_He, 'k:', label=r"$X_{\rm e, He}$")
    ax.set_ylabel(r"${\rm Ionization fraction,}~X_{\rm e}$")
    ax.set_xlabel(r"${\rm Redshift,}~z$")
    ax.set_xlim(500, 2500)
    ax.legend(loc=2, frameon=False)
    ax.tick_params(direction="inout", axis="both")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig(results_dir / 'testing_recfast.pdf')
    return 

def drag_epoch_testing(p):
    zarr, Xe_H, Xe_He, Xe ,TM = recfast.Xe_frac(p.Gamma_P, p.Tcmb, p.Omega_c, p.Omega_b, p.Omega_lambda, p.Omega_k, np.sqrt(p.hsquared), p.Nnu_massless, p.F, p.fDM)
    xe = interp1d(zarr, Xe, kind='cubic')
    zarr = np.array(zarr)
    R_div_a = 3 * p.rho_b_0 / (4. * p.rho_gamma_0)
    Hubble = np.vectorize(lambda x: H(x, p))
    def integrand(x):
        return sigma_T * p.rho_b_0 / mH * (1 - p.Gamma_P) / R_div_a * xe(x) / Hubble(x) * (1. + x) ** 3
    fig, ax = plt.subplots(1, 1)
    ax.plot(zarr, integrand(zarr))
    ax.set_xlim(0, 800)
    ax.set_yscale('log')
    fig.savefig("integrand_testing.pdf")

    print(quad(integrand, 1, 800, limit=50))


    return 

def main(argv):
    del argv 

    results_dir = Path(FLAGS.results_dir).absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if FLAGS.mode == "PlotBAOData":
        PlotBAOData(results_dir)

    if FLAGS.mode == "RECFAST":
        PlotRECFAST(results_dir)

    if FLAGS.mode == "testing":
        pars = Params(omega_b=0.0225,omega_c=0.12,H0=67.0,Nnu_massive=1.0,Nnu_massless=2.046,mnu=0.06, Omega_k=0.,Tcmb=2.7255, w=-1, Gamma_P=0.24, F=1.14, fDM=0.)
        print(z_drag(pars))
        print(r_drag(pars) / Mpc)
        #drag_epoch_testing(pars)

if __name__ == "__main__":
    flags.DEFINE_enum(
        "mode", 
        "PlotBAOData", 
        ["PlotBAOData",
        "RECFAST",
        "testing"], 
        "Which mode to run in.")
    flags.DEFINE_string(
        "results_dir", 
        "./results/background", 
        "Directory to write results.")
    app.run(main)