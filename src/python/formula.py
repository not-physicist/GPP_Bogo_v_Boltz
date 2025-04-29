###########################################
# contains formulas as functions
###########################################
import numpy as np

def get_f(k, a_e_rh, H_e, Γ):
    """
    calculate current GW frequency from 
    k = k/a_e H_e 
    H_e and Γ are in planck mass

    returns frequency in hertz
    """
    return 1.80e10 * k * a_e_rh * H_e * Γ**(-1/2)

def get_Ω_gw0(ρ, a_e_rh, H_e, Γ):
    """
    calculate current GW energy parameter
    ρ = a^4 ρ / a_e^4 H_e^4
    """
    return 3.83e-14 * ρ * a_e_rh**4 * H_e**4 * Γ**(-2)

def get_f_ana(k, Hₑ, mᵩ, Γ):
    """
    f = |βₖ|^2
    using analytical Boltzmann results
    k is k/a_e H_e
    """
    H_m = Hₑ / mᵩ
    # H_m = 1
    ex = np.exp(-4*Γ/(3*Hₑ) * ( (k*H_m)**(3/2) - 1) )
    f = 9*np.pi / 16 * H_m**(-3/2) * k**(-9/2) * ex
    return f


def get_f_exact_boltz(k, a, ρ_ϕ, H, m_ϕ, aₑ, Hₑ):
    """
    f = |βₖ|^2
    using exact Boltzmann and interpolation
    k is k/a_e H_e
    """
    # convert to k / m_ϕ
    # k_new = k * Hₑ / m_ϕ * aₑ
    # print(np.log(k_new)[0])
    # print(a/aₑ*m_ϕ/Hₑ)
    print(m_ϕ/Hₑ)

    # n^2 / H
    n2_H = (ρ_ϕ / m_ϕ)**2 / H
    # n2_H = (3*Hₑ**2*(a/aₑ)**(-3)/m_ϕ)**2 / (Hₑ * (a/aₑ)**(-3/2))
    
    # interpolate a/a_e == k / (a_e m_ϕ)
    n2_H_new = np.interp(k, a/aₑ*m_ϕ/Hₑ, n2_H)
    f =  np.pi / 16 / m_ϕ * n2_H_new

    return f
