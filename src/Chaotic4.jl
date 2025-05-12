"""
Chaotic n=4 inflation
"""

module Chaotic4 

using ..Commons 
using ..EOMs
using ..PPs 
using ..TModel 

using StaticArrays, Logging, Printf, Serialization, NPZ, NumericalIntegration, LinearInterpolations

const MODEL_NAME = "Chaotic4"
const MODEL_DATA_DIR = "data/$MODEL_NAME/"

function get_V(ϕ, l)
    return l * ϕ^4 
end 

function get_dV(ϕ, l)
    return l * 4 * ϕ^3
end

"""
get the potential parameter l from desired n_s and r by matching to T Model 
"""
function get_l(r)
    # ACT best fit
    ns = 0.974
    # dont care about ϕᵢ, set to 0.0
    tmodel = TModel.TModels(4, ns, r, 0.0)
    return TModel.get_λ(tmodel)
end

function save_eom(l, Γ, data_dir)
    ϕᵢ = 5.0 
    dVᵢ = get_dV(ϕᵢ, l)
    Vᵢ = get_V(ϕᵢ, l)
    dϕᵢ = get_dϕ_SR(dVᵢ, Vᵢ, 1.0)

    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 1e9)
    _V(x) = get_V(x, l)
    _dV(x) = get_dV(x, l)
    p = (_V, _dV, Γ)

    EOMs.save_all(u₀, tspan, p, data_dir)
    return nothing
end

test_save_eom() = save_eom(get_l(0.001), 1e-12, MODEL_DATA_DIR * "test/")

end

