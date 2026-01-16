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
    # @show model.α
    return model.V₀ / ((6*model.α)^(model.n/2))
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
function save_eom(ϕᵢ, r, T, n, data_dir::String)
    # mkpath(data_dir)

    model = TModels(n, 0.974, r, ϕᵢ)
    @info Commons.dump_struct(model)
    # @info data_dir
    save_model_data(model, data_dir * "model.dat")

    # initial conditions
    # ϕᵢ *= model.ϕₑ
    dϕᵢ = get_dϕ_SR(get_dV(ϕᵢ, model), get_V(ϕᵢ, model))

    # @info "Initial conditions are: ", ϕᵢ, dϕᵢ
    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 1e12)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    _m_eff(x) = get_m_eff(x, model)
    # α = 3 * (n-2)/(n+2)
    p = (_V, _dV, T, n)
    if n == 2 
        dtmax = 1/get_m_eff(0.0, model)/200
    elseif n == 4
        dtmax = 1/(10*sqrt(get_λ(model))*2*sqrt(3)) / 100000
    elseif n == 6
        dtmax = 1/(10*sqrt(get_λ(model))*2*sqrt(3)) / 500000
    end
    @show dtmax
    
    EOMs.save_all(u₀, tspan, p, data_dir, _m_eff, dtmax)

    return nothing
end

# m ~ 1e-5 mpl 
# save_eom_benchmark() = save_eom(3.6, 0.0045, 1e-7, 2)
# save_f_benchmark() = PPs.save_all(100, MODEL_DATA_DIR*"0.0045/")

function save_single(ϕᵢ, r, T, n, num_k, k_min=-2, k_max=2)
    data_dir = @sprintf "%s-n=%i/r=%.1e-T=%.1e/" MODEL_DATA_DIR n r T
    @info data_dir
    mkpath(data_dir)
    @info "Model parameter (in GeV): " r, T
    
    # save_eom(ϕᵢ, r, T, n, data_dir)
    if !isnothing(num_k)
        # PPs.save_all(num_k, data_dir, k_min, k_max)
        if n == 2
            Boltzmann.save_all(num_k*5, data_dir, :quadratic, k_min, k_max, true)
        elseif n == 4
            PPs.save_all_ana(num_k*5, data_dir, :quartic, k_min, k_max)
            # Boltzmann.save_all(num_k*5, data_dir, :quartic, k_min, k_max, true)
        elseif n == 6
            PPs.save_all_ana(num_k*5, data_dir, :sextic, k_min, k_max)
            # Boltzmann.save_all(num_k*5, data_dir, :sextic, k_min, k_max, true)
        end
    end
end

# function save_all_spec()
#     r_array = [0.0045]
#     Γ_array = logspace(-8, -6, 3)
#     num_k = 100
#     n = 2
#
#     for r in r_array 
#         for Γ in Γ_array
#             save_single(ϕᵢ, r, Γ, n, num_k)
#         end 
#     end 
#     return nothing
# end

end
