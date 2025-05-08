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

using StaticArrays, NPZ, Logging, Printf
# using JLD2

# global constant
const MODEL_NAME="TModel"
# not complete dir!
const MODEL_DATA_DIR="data/$MODEL_NAME/"

function get_V(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.V₀ * tanh(x)^(model.n)
end

function get_dV(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.n / sqrt(6*model.α) * model.V₀ * sech(x)^2 * tanh(x)^(model.n-1)
end

"""
get parameters for (Taylored) monomial potential 
"""
function get_λ(model)
    return model.V₀ / ((6*model.α)^(model.n))
end

"""
get inflaton mass at oscillation
"""
function get_m(model)
    return sqrt(2*get_λ(model))
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
function save_eom(ϕᵢ::Float64, r::Float64, Γ::Float64, n::Int64, data_dir::String)
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
    tspan = (0.0, 1e7)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV, Γ)
    
    EOMs.save_all(u₀, tspan, p, data_dir)

    return nothing
end

# m ~ 1e-5 mpl 
# save_eom_benchmark() = save_eom(3.6, 0.0045, 1e-7, 2)
# save_f_benchmark() = PPs.save_all(100, MODEL_DATA_DIR*"0.0045/")

function save_all_spec()
    r_array = [0.0045]
    Γ_array = logspace(-8, -6, 3)
    num_k = 100

    for r in r_array 
        for Γ in Γ_array
            data_dir = @sprintf "%sr=%.1e-Γ=%.1e/" MODEL_DATA_DIR r Γ
            mkpath(data_dir)
            @info "Model parameter (in GeV): " r, Γ

            save_eom(3.6, r, Γ, 2, data_dir)
            PPs.save_all(num_k, data_dir)
        end 
    end 
    return nothing
end

end
