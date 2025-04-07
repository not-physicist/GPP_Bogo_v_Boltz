"""
Chaotic inflation
"""

module Chaotic2

using ..EOMs

using StaticArrays, Logging, Printf

const MODEL_NAME = "Chaotic2"
const MODEL_DATA_DIR = "data/$MODEL_NAME"

function get_V(ϕ, m)
    return m^2 * ϕ^2 / 2.
end 

function get_dV(ϕ, m)
    return m^2 * ϕ
end

function save_eom(m::Float64, Γ::Float64, data_dir::String=MODEL_DATA_DIR)
    mkpath(data_dir)
    
    @info data_dir 
    @info m, Γ

    # initial conditions
    ϕᵢ = 10 
    # ignore ddϕ and Γ in EOM, take a=1
    # conformal Hubble
    Hᵢ = sqrt(get_V(ϕᵢ, m)/3.)
    dVᵢ = get_dV(ϕᵢ, m)
    dϕᵢ = - dVᵢ / (2*Hᵢ)

    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 5e4)
    _V(x) = get_V(x, m)
    _dV(x) = get_dV(x, m)
    p = (_V, _dV, Γ)
    sol = EOMs.solve_eom(u₀, tspan, p, nothing)
end

save_eom_test() = save_eom(1e-5, 1e-6)

end
