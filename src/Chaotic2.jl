"""
Chaotic inflation
"""

module Chaotic2

using ..Commons
using ..EOMs
using ..PPs

using StaticArrays, Logging, Printf, Serialization, NPZ, NumericalIntegration, LinearInterpolations

const MODEL_NAME = "Chaotic2"
const MODEL_DATA_DIR = "data/$MODEL_NAME/"

function get_V(ϕ, m)
    return m^2 * ϕ^2 / 2.
end 

function get_dV(ϕ, m)
    return m^2 * ϕ
end

function save_eom(m::Float64, Γ::Float64, data_dir)
    # initial conditions
    ϕᵢ = 6.0
    # ignore ddϕ and Γ in EOM, take a=1
    # conformal Hubble
    # Hᵢ = sqrt(get_V(ϕᵢ, m)/3.)
    dVᵢ = get_dV(ϕᵢ, m)
    Vᵢ = get_V(ϕᵢ, m)
    dϕᵢ = get_dϕ_SR(dVᵢ, Vᵢ, 1.0)

    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 1e7)
    _V(x) = get_V(x, m)
    _dV(x) = get_dV(x, m)
    p = (_V, _dV, Γ)

    EOMs.save_all(u₀, tspan, p, data_dir)
    # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    
    return nothing
end

#= 
function save_f(num_k=100, data_dir=MODEL_DATA_DIR)    
    PPs.save_all(num_k, eom, data_dir)

    # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    return nothing
end
=#

function save_all_spec()
    m_array = logspace(-4, -6, 3)
    # m_array = [1e-5]
    Γ_m_array = logspace(-3, -1, 3)
    num_k = 100
    # @show m_array, Γ_array, logspace(-3, -1, 3)
    
    for m in m_array
        for Γ_m in Γ_m_array
            Γ = Γ_m * m
            # data_dir = MODEL_DATA_DIR * "m=$(@sprintf("%.2e", m))" * "-Γ=$(@sprintf("%.2e", Γ)/"
            data_dir = @sprintf "%sm=%.1e-Γ=%.1e/" MODEL_DATA_DIR m Γ
            mkpath(data_dir)
            # @info "data_dir = $(data_dir)" 
            @info "Model parameter (in GeV): " m, Γ

            save_eom(m, Γ, data_dir)
            PPs.save_all(num_k, data_dir)
        end
    end
end

save_eom_test() = save_eom(1e-5, 1e-7, MODEL_DATA_DIR * "test/")
save_f_test() = PPs.save_all(100, MODEL_DATA_DIR * "test/")

end
