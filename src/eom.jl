"""
solve equation of motion for inflaton field 
including inflaton decay and radiation energy density
"""

module EOMs 

using StaticArrays, OrdinaryDiffEq, Logging

function _get_ρ_ϕ(ϕ, dϕ, a, V)
    return dϕ^2 / (2*a^2) + V(ϕ)
end

function _get_H2_conf(ϕ, dϕ, a, ρ_r, V)
    ρ_ϕ = _get_ρ_ϕ(ϕ, dϕ, a, V)
    return a^2 * (ρ_r + ρ_ϕ) / 3.
end

"""
Friedmann equation, EOM of inflaton field 
and radiation energy density

In conformal time
"""
function _get_f(u, p, t)
    ϕ = u[1]
    dϕ = u[2]
    a = u[3]
    ρ_r = u[4]

    get_V = p[1]
    get_dV = p[2]
    Γ = p[3]

    ρ_ϕ = _get_ρ_ϕ(ϕ, dϕ, a, get_V)
    # conformal Hubble
    H = sqrt(abs(_get_H2_conf(ϕ, dϕ, a, ρ_r, get_V)))

    return SA[dϕ, 
              - (2*H + a*Γ)*dϕ - a^2 * get_dV(ϕ), 
              a*H, 
              -4*H*ρ_r + a*Γ*ρ_ϕ]
end

"""
function for isoutofdomain
return true when H is sqrt of some negative number
"""
function _H_neg(u, p, t)
    H2 = _get_H2_conf(u[1], u[2], u[3], u[4], p[1])
    if H2 < 0.0 
        return true 
    else
        return false
    end
end

"""
Solve the EOMs given the parameters and initial conditions
"""
function solve_eom(u₀::SVector{4, Float64},
                   tspan::Tuple{Float64, Float64},
                   p::Tuple{Function, Function, Float64},
                   cb)

    prob = ODEProblem(_get_f, u₀, tspan, p)
    sol = solve(prob, Tsit5(), isoutofdomain=_H_neg, callback=cb)
    
    # return sol.t, sol[1, :], sol[2, :], sol[3, :], sol[4, :]
    return sol
end

end
