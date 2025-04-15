###########################################
# contains formulas as functions
###########################################

def get_f(k, a_e_rh, H_e, Γ):
    """
    calculate current GW frequency from 
    k = k/a_e H_e 
    H_e and Γ are in planck mass

    returns frequency in hertz
    """
    return 1.23e10 * k * a_e_rh * H_e * Γ**(-1/2)

def get_Ω_gw0(ρ, a_e_rh, H_e, Γ):
    """
    calculate current GW energy parameter
    ρ = a^4 ρ / a_e^4 H_e^4
    """
    return 8.40e-15 * ρ * a_e_rh**4 * H_e**4 * Γ**(-2)
