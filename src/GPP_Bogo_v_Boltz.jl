"""
Program in reduced planck unit
"""
module GPP_Bogo_v_Boltz

# export SmallFields
# export TModes

include("common.jl")
include("boltz.jl")
include("pp.jl")
include("eom.jl")

include("TModel/TModel.jl")
using .TModel
export TModel

include("Chaotic2.jl")
using .Chaotic2
export Chaotic2 

include("Chaotic4.jl")
using .Chaotic4
export Chaotic4

include("Chaotic6.jl")
using .Chaotic6 
export Chaotic6 

function save_all_TModel(ϕᵢ, r, Γ, num_k)
    TModel.save_single(ϕᵢ, r, Γ, 2, num_k)
    TModel.save_single(ϕᵢ, r, Γ, 4, num_k)
    TModel.save_single(ϕᵢ, r, Γ, 6, num_k)
end

function save_all_TModel()
    num_k = 100
    r = 0.01
    Γ = 1e-10

    TModel.save_single(5.0, r, Γ, 2, num_k)
    TModel.save_single(5.5, r, Γ, 4, num_k)
    TModel.save_single(6.5, r, Γ, 6, num_k)
end

function save_all_Chaotic(r, Γ, num_k)
    m = Chaotic2.get_m(r)
    Chaotic2.save_single(m, Γ, num_k)
    # Chaotic4.save_single(r, Γ, num_k)
    # Chaotic6.save_single(r, Γ, num_k)
end
save_all_Chaotic() = save_all_Chaotic(0.01, 1e-8, 100)

end
