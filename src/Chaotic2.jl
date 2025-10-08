"""
Chaotic inflation
"""

module Chaotic2

using ..Commons
using ..EOMs
using ..PPs
using ..TModel 
using ..Boltzmann

using StaticArrays, Logging, Printf, Serialization, NPZ, NumericalIntegration, LinearInterpolations

const MODEL_NAME = "Chaotic2"
const MODEL_DATA_DIR = "data/$MODEL_NAME/"

"""
get the potential parameter l from desired n_s and r by matching to T Model 
"""
function get_m(r)
    # ACT best fit
    ns = 0.974
    # dont care about ϕᵢ, set to 0.0
    tmodel = TModel.TModels(2, ns, r, 0.0)
    m = sqrt(2 * TModel.get_λ(tmodel))
    return m
end

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
    tspan = (0.0, 1e11)
    _V(x) = get_V(x, m)
    _dV(x) = get_dV(x, m)
    α = 0
    p = (_V, _dV, Γ, α)
    dtmax = 2*π/m / 100

    EOMs.save_all(u₀, tspan, p, data_dir, dtmax)
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

function save_single(m, Γ, num_k)
    data_dir = @sprintf "%sm=%.1e-Γ=%.1e/" MODEL_DATA_DIR m Γ
    mkpath(data_dir)
    # @info "data_dir = $(data_dir)" 
    @info "Model parameter (in GeV): " m, Γ

    save_eom(m, Γ, data_dir)
    PPs.save_all(num_k, data_dir)
    # Boltzmann.save_all(num_k, data_dir, :quadratic)
end

function save_all_spec()
    # m_array = logspace(-4, -6, 3)
    m_array = [1e-5]
    Γ_m_array = logspace(-2, -1, 3)
    num_k = 100
    # @show m_array, Γ_array, logspace(-3, -1, 3)
    
    for m in m_array
        for Γ_m in Γ_m_array
            Γ = Γ_m * m
            save_single(m, Γ, num_k)
        end
    end
end

# save_eom_test() = save_eom(1e-5, 1e-8, MODEL_DATA_DIR * "test/")
# save_f_test() = PPs.save_all_every(MODEL_DATA_DIR * "test/")
# save_f_test() = PPs.save_all(100, MODEL_DATA_DIR * "test/")

end
