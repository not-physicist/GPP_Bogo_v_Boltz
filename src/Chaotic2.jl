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

function save_eom(m::Float64, Γ::Float64, data_dir::String=MODEL_DATA_DIR)
    mkpath(data_dir)
    
    @info "data_dir = $(data_dir)" 
    @info "Model parameter (in GeV): " m, Γ

    # initial conditions
    ϕᵢ = 4
    # ignore ddϕ and Γ in EOM, take a=1
    # conformal Hubble
    Hᵢ = sqrt(get_V(ϕᵢ, m)/3.)
    dVᵢ = get_dV(ϕᵢ, m)
    dϕᵢ = - dVᵢ / (2*Hᵢ)

    u₀ = SA[ϕᵢ, dϕᵢ, 1.0, 0.0]
    tspan = (0.0, 1e6)
    _V(x) = get_V(x, m)
    _dV(x) = get_dV(x, m)
    p = (_V, _dV, Γ)

    sol = @time EOMs.solve_eom(u₀, tspan, p)

    get_end(sol)
    # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

    # τ = sol.t 
    ϕ = sol[1, :]
    dϕ = sol[2, :]
    a = sol[3, :]
    ρ_r = sol[4, :]
    # @show ϕ[end-10:end]
    
    ρ_ϕ = @. get_ρ_ϕ(ϕ, dϕ, a, x->get_V(x, m))
    ρ_tot = ρ_r + ρ_ϕ

    a_end, H_end = get_end(sol)
    # @show a_end, H_end
    
    mkpath(data_dir)
    # serialize for Bogoliubov computation
    serialize(data_dir * "ode.dat", (sol, a_end, H_end))
    # @show typeof(sol)

    app_a = @. (4 * sol[3, :]^2 * get_V(sol[1, :], m) - sol[2, :]^2) / 6.0

    # mainly for plotting
    npzwrite(data_dir * "ode.npz", Dict(
    "tau" => sol.t,
    "phi" => ϕ,
    "a" => a,
    # "epsilon1" => get_end(sol),
    "app_a" => app_a,
    "Omega_r" => ρ_r ./ ρ_tot,
    "Omega_phi" => ρ_ϕ ./ ρ_tot,
    ))
    
    return nothing
end

function save_f(m::Float64, num_k=100, data_dir=MODEL_DATA_DIR)
    @info "Reading $(MODEL_DATA_DIR * "ode.dat")"
    sol, a_e, H_e = deserialize(data_dir * "ode.dat")
    # @show a_e * H_e, m

    # conformal Hubble
    get_H(t) = @. sol(t, Val{1}, idxs=3) / sol(t, idxs=3)
    
    k = @. logspace(log10(2), 2.0, num_k) * a_e * H_e
    # k = @. [1.5] * a_e * H_e
    
    # a''/a according to second Friedmann
    app_a = @. (4 * sol[3, :]^2 * get_V(sol[1, :], m) - sol[2, :]^2) / 6.0
    # (a''/a)'
    app_a_p = diff(app_a[1:end-1]) ./ diff(sol.t[1:end-1])

    # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

    # @show k .^ 2
    # @show app_a[1:200:end]
    # @show app_a_p[end-10:end]
    # @show get_H(sol.t[1:100:end])

    n, ρ, error = PPs.get_f(k, sol, app_a, app_a_p)
    # @show n, ρ, error
    
    npzwrite(data_dir * "spec.npz", Dict(
        "k" => k ./ (a_e*H_e),
        "n" => abs.(n),
        "rho" => abs.(ρ ./ (a_e*H_e)),
        "error" => error
    ))
    return nothing
end

save_eom_test() = save_eom(1e-5, 1e-7)
save_f_test() = save_f(1e-5)

end
