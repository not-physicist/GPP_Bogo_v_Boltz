"""
Model for T-model α attractor inflation potential
"""
module TModel

# submodules
include("ModelData.jl")
using .ModelDatas

using ..EOMs
using ..Commons
using ..PPs
using ..Boltzmann

using StaticArrays, NPZ, Logging, Printf
# using JLD2

# global constant
const MODEL_NAME="TModel"
# not complete dir!
const MODEL_DATA_DIR="data/$MODEL_NAME"

function get_V(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.V₀ * tanh(x)^(model.n)
end

function get_dV(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.n / sqrt(6*model.α) * model.V₀ * sech(x)^2 * tanh(x)^(model.n-1)
end

"""
(time dependent) inflaton effective mass
"""
function get_m_eff(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    ddV = model.V₀*model.n/(6*model.α) * tanh(x)^(model.n-2) * (1 - 4*tanh(x)^2 + 3*tanh(x)^2)
    return sqrt(ddV)
end

"""
get parameters for (Taylored) monomial potential 
"""
function get_λ(model)
    @show model.α
    return model.V₀ / ((6*model.α)^(model.n))
end

#=
"""
dϕ = dϕ/dτ at slow roll trajectory in conformal time
"""
function get_dϕ_SR(ϕ::Real, model::TMode, a::Real=1.0)
    return - a * get_dV(ϕ, model) / sqrt(3 * get_V(ϕ, model))
end

"""
define inflationary scale like this
"""
function get_Hinf(model::TMode)
    return √(model.V₀ / 3.0)
end
=#

"""
ϕᵢ: in unit of ϕₑ (field value at end of slow roll)
init_time_mul: initial (conformal) time multiplicant; needs to make the simulation run longer for large r 
"""
function save_eom(ϕᵢ, r, Γ, n, data_dir::String)
    # mkpath(data_dir)

    model = TModels(n, 0.965, r, ϕᵢ)
    # @info Commons.dump_struct(model)
    # @info data_dir
    save_model_data(model, data_dir * "model.dat")

    # initial conditions
    # ϕᵢ *= model.ϕₑ
    dϕᵢ = get_dϕ_SR(get_dV(ϕᵢ, model), get_V(ϕᵢ, model))

    # @info "Initial conditions are: ", ϕᵢ, dϕᵢ
    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 1e10)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    _m_eff(x) = get_m_eff(x, model)
    α = 3 * (n-2)/(n+2)
    p = (_V, _dV, Γ, α)
    dtmax = 2*π/_m_eff(0.0) / 100
    
    EOMs.save_all(u₀, tspan, p, data_dir, _m_eff, dtmax)

    return nothing
end

# m ~ 1e-5 mpl 
# save_eom_benchmark() = save_eom(3.6, 0.0045, 1e-7, 2)
# save_f_benchmark() = PPs.save_all(100, MODEL_DATA_DIR*"0.0045/")

function save_single(ϕᵢ, r, Γ, n, num_k)
    data_dir = @sprintf "%s-n=%i/r=%.1e-Γ=%.1e/" MODEL_DATA_DIR n r Γ
    @info data_dir
    mkpath(data_dir)
    @info "Model parameter (in GeV): " r, Γ

    save_eom(ϕᵢ, r, Γ, n, data_dir)
    PPs.save_all(num_k, data_dir)
    # Boltzmann.save_all(num_k, data_dir)
end

function save_all_spec()
    r_array = [0.0045]
    Γ_array = logspace(-8, -6, 3)
    num_k = 100
    n = 2

    for r in r_array 
        for Γ in Γ_array
            save_single(ϕᵢ, r, Γ, n, num_k)
        end 
    end 
    return nothing
end

end
