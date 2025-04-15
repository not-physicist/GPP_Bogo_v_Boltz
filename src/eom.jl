"""
solve equation of motion for inflaton field 
including inflaton decay and radiation energy density
"""

module EOMs 

using ..Commons

using StaticArrays, OrdinaryDiffEq, Logging, NPZ, Serialization

"""
Friedmann equation, EOM of inflaton field 
and radiation energy density

In conformal time
"""
function _get_f(u, p, t)
    ϕ = u[1]
    dϕ = u[2]
    a = u[3]
    ρ_r = u[4]

    get_V = p[1]
    get_dV = p[2]
    Γ = p[3]

    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, get_V)
    # conformal Hubble
    H = sqrt(abs(get_H2_conf(u..., get_V)))

    return SA[dϕ, 
              - (2*H + a*Γ)*dϕ - a^2 * get_dV(ϕ), 
              a*H, 
              -4*H*ρ_r + a*Γ*ρ_ϕ]
end

"""
function for isoutofdomain
return true when H is sqrt of some negative number
"""
function _H_neg(u, p, t)
    H2 = get_H2_conf(u..., p[1])
    if H2 < 0.0 
        return true 
    else
        return false
    end
end

"""
Solve the EOMs given the parameters and initial conditions
"""
function solve_eom(u₀::SVector{4, Float64},
                   tspan::Tuple{Float64, Float64},
                   p::Tuple{Function, Function, Float64})
    # callback: terminate at ρ_ϕ / ρ_tot = 1e-10
    _Omega_ϕ(u) = get_ρ_ϕ(u[1], u[2], u[3], p[1]) / (u[4] + get_ρ_ϕ(u[1], u[2], u[3], p[1])) 
    condition(u, t, integrator) = ( _Omega_ϕ(u) <= 1e-5)
    # condition(u, t, integrator) = ( u[3] >= 100)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    prob = ODEProblem(_get_f, u₀, tspan, p)
    sol = solve(prob, Tsit5(), isoutofdomain=_H_neg, reltol=1e-12, abstol=1e-12, callback=cb)
    
    return get_EOMData(sol, p[1], p[3])
end

"""
energy density of inflaton field
in conformal time (dϕ = dϕdτ)
"""
function get_ρ_ϕ(ϕ, dϕ, a, V)
    return dϕ^2 / (2*a^2) + V(ϕ)
end

"""
conformal Hubble squared  
"""
function get_H2_conf(ϕ, dϕ, a, ρ_r, V)
    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, V)
    return a^2 * (ρ_r + ρ_ϕ) / 3.
end

"""
from ODEsolution get ODEData
"""
function get_EOMData(sol::SciMLBase.ODESolution, _V::Function, Γ::Float64)
    τ = sol.t 
    ϕ = sol[1, :]
    dϕ = sol[2, :]
    a = sol[3, :]
    ρ_r = sol[4, :]
    # @show ϕ[end-10:end]
    
    ρ_ϕ = @. get_ρ_ϕ(ϕ, dϕ, a, _V)
    ρ_tot = ρ_r + ρ_ϕ
    # normal Hubble
    H = @. sqrt(ρ_tot / 3.0)
    # a''/a according to second Friedmann
    app_a = @. (4 * a^2 * _V(ϕ) - dϕ^2) / 6.0
    # (a''/a)'
    app_a_p = diff(app_a[1:end-1]) ./ diff(τ[1:end-1])
    Ω_r = @. ρ_r/(3*H^2)
    Ω_ϕ = @. ρ_ϕ/(3*H^2)

    # @show size(ϕ), size(dϕ)
    # @show size(app_a), size(app_a_p)
    # @show size(H), size(ρ_r), size(ρ_ϕ)

    a_e, H_e = get_end(sol)
    # reheating scale factor
    dec_index = findfirst(x -> x <= Γ, H)
    a_rh = a[dec_index]
    # @show log(a_rh)

    # now need to discard the last two elements
    # as app_a_p miss these
    return EOMData(τ[1:end-2], ϕ[1:end-2], dϕ[1:end-2], 
            a[1:end-2], app_a[1:end-2], app_a_p,
            H[1:end-2], Ω_r[1:end-2], Ω_ϕ[1:end-2],
            a_e, a_rh, H_e)
end


"""
struct to store the ODE data;
note that they may have different length (due to the derivatives)
"""
struct EOMData{V<:Vector, F<:Real}
    τ::V
    ϕ::V
    dϕ::V
    a::V
    app_a::V
    app_a_p::V
    H::V
    Ω_r::V 
    Ω_ϕ::V

    aₑ::F
    a_rh::F  # scale factor at H = Γ
    Hₑ::F
end

#=
"""
read ODE solution stored in data/ode.npz
"""
function read_ode(data_dir::String)::ODEData
    # maybe a try catch clause here; not sure if necessary
    fn = data_dir * "ode.npz"
    data = npzread(fn)
    #  fn = data_dir * "ode.jld2"
    #  data = load(fn)

    τ = data["tau"]
    ϕ = data["phi"]
    dϕ = data["phi_d"]
    a = data["a"]
    app_a = data["app_a"]
    H = data["H"]
    err = data["err"]
    aₑ = data["a_end"]
    Hₑ = data["H_end"]
    return ODEData(τ, ϕ, dϕ, a, app_a, H, err, aₑ, Hₑ)
end
=#

function get_end(sol::SciMLBase.ODESolution)
    _a(t) = sol(t, Val{0}, idxs=3)
    _ap(t) = sol(t, Val{1}, idxs=3)
    _app(t) = sol(t, Val{2}, idxs=3)

    _ϵ₁(t) = @. 2 - _app(t) * _a(t) / _ap(t)^2
    # normal Hubble
    _H(t) = @. _ap(t) / _a(t)^2
   
    end_i = findfirst(x -> x >= 1.0, _ϵ₁(sol.t))
    # @show _ϵ₁(sol.t[end_i]), log(_a(sol.t[end_i])), _H(sol.t[end_i])
    a_end = _a(sol.t[end_i])
    H_end = _H(sol.t[end_i])

    # return _ϵ₁(sol.t)
    return a_end, H_end
end

"""
save the quantities in EOMData into npz file
mainly for plotting
"""
function save_eom_npz(eom::EOMData, data_dir)
    npzwrite(data_dir * "eom.npz", Dict(
    "tau" => eom.τ,
    "phi" => eom.ϕ,
    "phi_d" => eom.dϕ,
    "a" => eom.a,
    "app_a" => eom.app_a,
    "app_a_p" => eom.app_a_p,
    "H" => eom.H,

    "Omega_r" => eom.Ω_r,
    "Omega_phi" => eom.Ω_ϕ,

    "a_e" => eom.aₑ,
    "a_rh" => eom.a_rh,
    "H_e" => eom.Hₑ
    ))
end

function save_all(u₀, tspan, p, data_dir)
    eom_data = @time solve_eom(u₀, tspan, p)

    mkpath(data_dir)
    # serialize for Bogoliubov computation
    serialize(data_dir * "eom.dat", eom_data)
    save_eom_npz(eom_data, data_dir)
    return nothing
end
end
