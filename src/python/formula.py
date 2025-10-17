###########################################
# contains formulas as functions
###########################################
gStar = 106.75

import numpy as np

#####################################################################

def get_T_rh(Γ):
    T = 1.41 * gStar**(-1/4) * Γ**(1/2)
    # print(T)
    return T

def get_f(k, a_e_rh, H_e, Γ):
    """
    calculate current GW frequency from 
    k = k/a_e H_e 
    H_e and Γ are in planck mass

    returns frequency in hertz
    """
    # return 1.80e10 * k * a_e_rh * H_e * Γ**(-1/2)
    return 18.8e9 * k * a_e_rh * H_e / get_T_rh(Γ)

def get_Ω_gw0(ρ, a_e_rh, H_e, Γ):
    """
    calculate current GW energy parameter
    ρ = a0^4 ρ / a_e^4 H_e^4
    """
    # return 3.83e-14 * ρ * a_e_rh**4 * H_e**4 * Γ**(-2)
    return 0.37 * ρ * a_e_rh**4 * (H_e/get_T_rh(Γ))**4

#####################################################################

def get_f_ana(k, Hₑ, mᵩ, Γ):
    """
    f = |βₖ|^2
    using analytical Boltzmann results
    k is k/a_e H_e
    """
    H_m = Hₑ / mᵩ
    print("H_e / m_phi = ", H_m)
    # H_m = 1
    ex = np.exp(-4*Γ/(3*Hₑ) * ( (k*H_m)**(3/2) - 1) )
    f = 9*np.pi / 64 * H_m**(-3/2) * k**(-9/2) * ex
    return f


def get_f_exact_boltz(k, a, ρ_ϕ, H, m_ϕ, aₑ, Hₑ):
    """
    f = |βₖ|^2
    using exact Boltzmann and interpolation
    k is k/a_e H_e
    """

    if np.isscalar(m_ϕ):
        # if only rest mass is given
        m_ϕ_end = m_ϕ
        # n^2 / H
        n2_H = ρ_ϕ**2 / m_ϕ**3 / H
        n2_H_new = np.where(k*Hₑ/m_ϕ_end > 1, 
                                          np.interp(k, a/aₑ*m_ϕ/Hₑ, n2_H),
                                          0)
    else:
        # m_ϕ is an array of effective masses
        m_ϕ_end = np.interp(aₑ, a, m_ϕ)
        # print("m_ϕ_end: ", m_ϕ_end, "; H_e: ", Hₑ, "; N_e: ", np.log(aₑ))
        n2_H = ρ_ϕ**2 / m_ϕ**2 / H
        n2_H = n2_H[:-1] / (m_ϕ[:-1] + np.diff(m_ϕ) / np.diff(np.log(a)))
        n2_H_new = np.where(k*Hₑ/m_ϕ_end > 1, 
                                          np.interp(k, (a/aₑ*m_ϕ/Hₑ)[:-1], n2_H),
                                          0)

    
    # interpolate a/a_e == k / (a_e m_ϕ)
    f =  np.pi / 64 * n2_H_new

    return f
