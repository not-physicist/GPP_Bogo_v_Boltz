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
const MODEL_DATA_DIR="data/$MODEL_NAME-"

function get_V(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.V₀ * tanh(x)^(model.n)
end

function get_dV(ϕ::Real, model)
    x = ϕ / (sqrt(6 * model.α))
    return model.n / sqrt(6*model.α) * model.V₀ * sech(x)^2 * tanh(x)^(model.n-1)
end

"""
get parameters for monomial potential (Taylored)
"""
function get_λ(model)
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
function save_eom(ϕᵢ::Float64, r::Float64, Γ::Float64, data_dir::String=MODEL_DATA_DIR*"$r/")
    mkpath(data_dir)

    model = TModel(1, 0.965, r, ϕᵢ)
    @info dump_struct(model)
    @info data_dir
    save_model_data(model, data_dir * "model.dat")

    # initial conditions
    ϕᵢ *= model.ϕₑ
    dϕᵢ = get_dϕ_SR(ϕᵢ, model)

    @info "Initial conditions are: ", ϕᵢ, dϕᵢ
    u₀ = SA[ϕᵢ, dϕdNᵢ, 1.0, 0.0]
    tspan = (0.0, 1e7)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV, Γ)
    
    EOMs.save_all(u₀, tspan, p, data_dir)

    return nothing
end
const dn_bm = "data/TModel-0.001-benchmark/"
save_eom_benchmark() = save_eom(1.7, 0.001, dn_bm)


function save_f(r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/";
                num_mᵪ::Int=20, num_k::Int=100)
    model = TModel(1, 0.965, r, NaN)
    @info dump_struct(model)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    # k = logspace(-2.0, 2.0, num_k) * ode.aₑ * mᵩ 
    k = logspace(log10(0.05), 2.0, num_k) * ode.aₑ * ode.Hₑ
    # mᵪ = SA[0.668724508653783 * mᵩ]
    mᵪ = SA[logspace(-2.0, log10(3.0), num_mᵪ).* mᵩ ...]
    # mᵪ = SA[logspace(log10(0.5), log10(3.0), num_mᵪ).* mᵩ ...]
    ξ = SA[0.0]
    # m3_2 = SA[collect(range(0.0, 2.0; length=num_m32)) * mᵩ ...]
    # m3_2 = SA[0.0] .* mᵩ
    m3_2 = SA[0.0, 0.01, 0.1, 0.2] .* mᵩ
    
    m2_eff_R(ode, mᵪ, ξ, m3_2) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_R, fn_suffix="_R")
     
    m2_eff_I(ode, mᵪ, ξ, m3_2) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_I, fn_suffix="_I")

    # PPs.save_each(data_dir * "nosugra/", mᵩ, ode, k, mᵪ, ξ, get_m2_no_sugra, solve_mode=true)
    return true
end


const dn_bm = "data/TModel-0.001-benchmark/"
save_eom_benchmark() = save_eom(1.7, 0.001, dn_bm)
save_f_benchmark() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=10, dn_bm)
save_f_benchmark2() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=100, dn_bm)

end
