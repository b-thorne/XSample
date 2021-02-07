import numpy as np 
import matplotlib.pyplot as plt 

def rho_gamma(rho_gamma_0, z):
    return rho_gamma_0 * (1 + z) ** 4

def rho_lambda(rho_lambda_0, w, z):
    return rho_lambda_0 * (1 + z) ** (3 * (1 + w))

def rho_c(rho_c_0, z):
    return rho_c_0 * (1 + z) ** 3

def rho_b(rho_b_0, z):
    return rho_b_0 * (1 + z) ** 
    
def rho_nu(rho_gamma_0, N_nu_massless, ):
    return rho_nu_massless(N_nu_massless, rho_gamma_0) + rho_nu_massive()

def rho_nu_massless(N_nu_massless, rho_gamma_0):
    return N_nu_massless * (7. / 8. * (4. / 11.) ** (4. / 3.)) * rho_gamma_0 * (1 + z) ** 4

def rho_nu_massive(mnu, z):
    if mnu == 0:
        return 0
    else:
        a = 1 / (1. + z)
        a_prime = 1e-7
        Tnu = T_gamma_0 * (4. / 11.) ** (1. / 3.)
        mTsquared = (mnu / (Tnu / a_prime)) ** 2
        def integrand(p):
            return p ** 2 * np.sqrt(((p * a_prime / a)) ** 2 + mTsquared) / (np.exp(np.sqrt(p ** 2 + mTsquared)) + 1)
    return 

if __name__ == "__main__":
