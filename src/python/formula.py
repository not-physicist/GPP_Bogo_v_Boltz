###########################################
# contains formulas as functions
###########################################
gStar = 106.75

import numpy as np
from scipy.special import hankel2 as h2
# from scipy.integrate import quadrature
from scipy import integrate

#####################################################################

def get_T_rh(Γ):
    T = 1.41 * gStar**(-1/4) * Γ**(1/2)
    # print(T)
    return T

def get_Γ(T):
    return 5 / (2*np.sqrt(3)) * np.pi/np.sqrt(30) * np.sqrt(gStar) * T**2

def get_f(k, a_e_rh, H_e, T):
    """
    calculate current GW frequency from 
    k = k/a_e H_e 
    H_e and Γ are in planck mass

    returns frequency in hertz
    """
    # return 1.80e10 * k * a_e_rh * H_e * Γ**(-1/2)
    print(a_e_rh * H_e / T)
    return 18.886e9 * k * a_e_rh * H_e / T

def get_Ω_gw0(ρ, a_e_rh, H_e, T):
    """
    calculate current GW energy parameter
    ρ = a0^4 ρ / a_e^4 H_e^4
    """
    # return 3.83e-14 * ρ * a_e_rh**4 * H_e**4 * Γ**(-2)
    print(a_e_rh**4 * (H_e/T)**4)
    return 0.4609 * ρ * a_e_rh**4 * (H_e/T)**4

#####################################################################

def get_f_ana(k, Hₑ, mᵩ, Γ):
    """
    f = |βₖ|^2
    using analytical Boltzmann results
    k is k/a_e H_e
    """
    H_m = Hₑ / mᵩ
    # print("H_e / m_phi = ", H_m)
    # H_m = 1
    ex = np.exp(-2*Γ/(Hₑ) * (k*H_m)**(3/2) )
    f = 9*np.pi / 64 * H_m**(-3/2) * k**(-9/2) * ex
    # print(np.interp(10, k, f))
    # print(ex)
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

def get_f_ana_slow(k, ω, μ):
    # k already unitless: k/a_e H_e
    qr_1 = -1 
    qr = 2/(1+3*ω)
    lr = qr_1 - 1/2
    mr = qr - 1/2
    xr = qr * k 
    yr = qr_1 * k
    # μ = 1.59
    # μ = 2
    
    # h2 is the second hankel function
    Q = h2(lr, yr) * h2(mr+1, xr) - h2(mr, xr) * h2(lr+1, yr)
    return np.absolute(np.pi/4 * np.sqrt(qr_1/qr+0.0J) * xr * Q)**2 * np.exp(- μ * k)

def _get_conf_H_fit(x, Δx, ω, Ht):
    # return [1/(((3*ω + 3)/4*np.tanh(i/Δx)+(3*ω-1)/4)*i/Ht + 1/Ht) if i > 0 else 1/(-i/Ht + 1/Ht) for i in x]
    return 1/(((3*ω + 3)/4*np.tanh(x/Δx)+(3*ω-1)/4)*x/Ht + 1/Ht)
    # return 1/(-x/Ht + 1/Ht)

def _get_transition_V(x, Δx, ω, Ht):
    # TODO: this formula is not exactly correct;
    # ω=-1 during inflation, but doesn't seem to make a big difference
    H = _get_conf_H_fit(x, Δx, ω, Ht)
    tanh = np.tanh(x/Δx)
    dH = -H**2 * (3*(ω+1)/4 * ((1-tanh**2)*x/Δx + tanh) + (3*ω-1)/4)
    return (dH + H**2)/Ht**2

def get_f_transition_int(k, ω, Δx, Ht):
    res, error = integrate.quad(lambda x: np.exp(-2.0j * k * x) * _get_transition_V(x, Δx, ω, Ht), -10000, +10000, limit=5000, complex_func=True)
    print(k, res, error)
    return (np.abs(res)/(2*k))**2

def get_f_transition_diffeq(k, ω, Δx, Ht):
    tau_max = 1000

    def _f(t, x):
        V = _get_transition_V(x[0], Δx, ω, Ht)
        return [x[1], -x[0]*(k**2 - V)]
    
    h0 = 1/np.sqrt(2*k*Ht)*np.exp(-1.0j*k*(-tau_max))
    x0 = [h0, -1.0j*k*h0]
    sol = integrate.solve_ivp(_f, [-tau_max, tau_max], x0)
    print(k, sol.y[:, -1])
    return 1/(2*k) * np.abs(k*sol.y[0, -1] - 1.0j*sol.y[1, -1])**2

def get_f_transition(k, Δτ, n):
    """
    k in a_e H_e unit
    """
    # return (np.pi*F_pk*Δτ**2*2*k/np.sinh(np.pi*k*Δτ))**2
    return np.abs(3*n/(4*(n+2)) * Δτ*np.pi / np.sinh(np.pi*k*Δτ) / k)**2
