"""
solve equation of motion for inflaton field 
including inflaton decay and radiation energy density
"""

module EOMs 

using ..Commons

using StaticArrays, OrdinaryDiffEq, Logging

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

    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, get_V)
    # conformal Hubble
    H = sqrt(abs(get_H2_conf(u..., get_V)))

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
    H2 = get_H2_conf(u..., p[1])
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
                   p::Tuple{Function, Function, Float64})
    # callback: terminate at ρ_ϕ / ρ_tot = 1e-10
    _Omega_ϕ(u) = get_ρ_ϕ(u[1], u[2], u[3], p[1]) / (u[4] + get_ρ_ϕ(u[1], u[2], u[3], p[1])) 
    condition(u, t, integrator) = ( _Omega_ϕ(u) <= 1e-10)
    # condition(u, t, integrator) = ( u[3] >= 100)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    prob = ODEProblem(_get_f, u₀, tspan, p)
    sol = solve(prob, Tsit5(), isoutofdomain=_H_neg, reltol=1e-12, abstol=1e-12, callback=cb)
    
    return sol
end

end
