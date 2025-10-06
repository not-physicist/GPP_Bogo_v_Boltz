using Pkg
Pkg.activate("./")

using GPP_Bogo_v_Boltz
r = 0.01 
Γ = 1e-9

TModel.save_single(4.0, r, Γ, 2, 50)
TModel.save_single(4.0, r, Γ, 4, 50)
TModel.save_single(4.0, r, Γ, 6, 50)
