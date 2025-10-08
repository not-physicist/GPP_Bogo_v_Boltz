"""
chaotic n=6 inflation
"""
module Chaotic6

using ..Commons 
using ..EOMs
using ..PPs 
using ..TModel 
using ..Boltzmann

using Printf, StaticArrays

const MODEL_NAME = "Chaotic6"
const MODEL_DATA_DIR = "data/$MODEL_NAME/"

function get_V(ϕ, l)
    return l * ϕ^6
end 

function get_dV(ϕ, l)
    return l * 6 * ϕ^5
end

"""
get the potential parameter l from desired n_s and r by matching to T Model 
"""
function get_l(r)
    # ACT best fit
    ns = 0.974
    # dont care about ϕᵢ, set to 0.0
    tmodel = TModel.TModels(6, ns, r, 0.0)
    return TModel.get_λ(tmodel)
end

function save_eom(l, Γ, data_dir)
    ϕᵢ = 9.0
    dVᵢ = get_dV(ϕᵢ, l)
    Vᵢ = get_V(ϕᵢ, l)
    dϕᵢ = get_dϕ_SR(dVᵢ, Vᵢ, 1.0)

    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    # tspan = (0.0, 1e9)
    tspan = (0.0, 1e10)
    _V(x) = get_V(x, l)
    _dV(x) = get_dV(x, l)

    α = 1.0
    p = (_V, _dV, Γ, α)

    # order of magnitude estimate for oscillation frequency
    ωStar = 10 * sqrt(l) * 2 * sqrt(3) 
    dtmax = 1/ωStar / 100000
    @show ωStar dtmax

    EOMs.save_all(u₀, tspan, p, data_dir, dtmax)
    return nothing
end

function save_single(r, Γ, num_k)
    data_dir = @sprintf "%sr=%.1e-Γ=%.1e/" MODEL_DATA_DIR r Γ 
    @info data_dir
    mkpath(data_dir)

    l = get_l(r)
    @info "Model parameter (in GeV): " l, Γ

    save_eom(l, Γ, data_dir)
    PPs.save_all(num_k, data_dir, -1, 2)
    # Boltzmann.save_all(num_k, data_dir, :sextic, 0, 2)
end

function save_all_spec()
    r_array = [0.0045]
    Γ_array = [1e-10]

    num_k = 100 

    for r in r_array 
        for Γ in Γ_array
            save_single(r, Γ, num_k)
            @printf "===============================================I am a separator============================================================\n"
        end
    end
end 

end
