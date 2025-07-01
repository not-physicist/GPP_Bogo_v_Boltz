"""
solve equation of motion for inflaton field 
including inflaton decay and radiation energy density
"""

module EOMs 

using ..Commons

using StaticArrays, OrdinaryDiffEq, Logging, NPZ, Serialization, NumericalIntegration, Peaks, LinearInterpolations

# TODO: want to have dtmax parameter passed into the module

"""
energy density of inflaton field
in conformal time (dϕ = dϕdτ)
"""
function get_ρ_ϕ(ϕ, dϕ, a, V, α)
    return dϕ^2 / (2*a^(2*α)) + V(ϕ)
end

"""
pressure of inflaton field
in conformal time (dϕ = dϕdτ)
"""
function get_p_ϕ(ϕ, dϕ, a, V, α)
    return dϕ^2 / (2*a^(2*α)) - V(ϕ)
end

"""
equation of state
"""
function get_eos(ϕ, dϕ, a, V, ρ_r, α)
    return (get_p_ϕ(ϕ, dϕ, a, V, α) + ρ_r/3)/(get_ρ_ϕ(ϕ, dϕ, a, V, α) + ρ_r)
end

"""
conformal Hubble squared 
w.r.t. the new time variable
"""
function get_H2_conf(ϕ, dϕ, a, ρ_r, V, α)
    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, V, α)
    return a^(2*α) * (ρ_r + ρ_ϕ) / 3.
end

"""
a''/a (in new time variable)
"""
function get_app_a(ϕ, dϕ, a, ρ_r, V, α)
    return a^(2*α)/3 * ((α-2)*dϕ^2/(2*a^(2*α)) + (α+1)*V(ϕ) + (α-1)*ρ_r)
end

"""
Hubble slow roll
"""
function get_ϵ1(ϕ, dϕ, a, ρ_r, V, α)
    return - get_app_a(ϕ, dϕ, a, ρ_r, V, α)/get_H2_conf(ϕ, dϕ, a, ρ_r, V, α) + α + 1
end 

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
    α = p[4]

    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, get_V, α)
    # conformal Hubble
    H = sqrt(abs(get_H2_conf(u..., get_V, α)))

    return SA[dϕ, 
             -((3-α)*H + a^(α)*Γ)*dϕ - a^(2*α) * get_dV(ϕ), 
              a*H, 
              -4*H*ρ_r + a^(α)*Γ*ρ_ϕ]
end

"""
function for isoutofdomain
return true when H is sqrt of some negative number
"""
function _H_neg(u, p, t)
    H2 = get_H2_conf(u..., p[1], p[4])
    if H2 < 0.0 
        return true 
    else
        return false
    end
end

"""
function for callback
"""
function _get_Omega_ϕ(u, p)
    return get_ρ_ϕ(u[1], u[2], u[3], p[1], p[4]) / (u[4] + get_ρ_ϕ(u[1], u[2], u[3], p[1], p[4]))
end

"""
Solve the EOMs given the parameters and initial conditions
"""
function solve_eom(u₀::SVector{4, Float64},
                   tspan::Tuple,
                   p::Tuple)
    # still inflation, no dissipation
    condition(u, t, integrator) = (abs(get_ϵ1(u..., p[1], p[4])) >= 1)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
    prob = ODEProblem(_get_f, u₀, tspan, (p[1], p[2], 0.0, p[4]))
    # sol = solve(prob, AutoVern9(Rodas5P()), isoutofdomain=_H_neg, reltol=1e-12, abstol=1e-12, callback=cb)
    sol = solve(prob, Vern9(), isoutofdomain=_H_neg, reltol=1e-12, abstol=1e-12, callback=cb, dtmax=1e4)
    
    # callback: terminate at ρ_ϕ / ρ_tot = 1e-5
    _Omega_ϕ(u) = _get_Omega_ϕ(u, p)
    condition2(u, t, integrator) = ( _Omega_ϕ(u) <= 1e-5)
    affect!(integrator) = terminate!(integrator)
    cb2 = DiscreteCallback(condition2, affect!)
    u₁ = SA[sol[1, end], sol[2, end], sol[3, end], sol[4, end]]
    tspan2 = (sol.t[end], tspan[2])
    prob = ODEProblem(_get_f, u₁, tspan2, p)
    sol2 = solve(prob, Vern9(), isoutofdomain=_H_neg, reltol=1e-12, abstol=1e-12, callback=cb2, dtmax=1e4, save_start=false)
    # @show sol[3, 1]
    
    Ω_ϕ_end = _Omega_ϕ(sol2.u[end])
    if Ω_ϕ_end >= 1e-5
        @warn "The EOM may not terminate properly. A longer simulation might be necessary! Ω_ϕ = %f" Ω_ϕ_end
    end

    η = vcat(sol.t[1:end-1], sol2.t)
    ϕ = vcat(sol[1, 1:end-1], sol2[1, :])
    dϕ = vcat(sol[2, 1:end-1], sol2[2, :])
    a = vcat(sol[3, 1:end-1], sol2[3, :])
    ρ_r = vcat(sol[4, 1:end-1], sol2[4, :])
    a_e = sol[3, end-1]
    
    return get_EOMData(η, ϕ, dϕ, a, ρ_r, p[1], p[3], p[4], a_e)
end

"""
from ODEsolution get ODEData
"""
function get_EOMData(η, ϕ, dϕ, a, ρ_r, V, Γ, α, a_e)
    # max_ind = argmaxima(ϕ)
    # @show η[max_ind]
    
    ρ_ϕ = @. get_ρ_ϕ(ϕ, dϕ, a, V, α)
    ρ_tot = ρ_r + ρ_ϕ
    # normal Hubble
    H = @. sqrt(ρ_tot / 3.0)
    # a''/a according to second Friedmann
    app_a = @. (4 * a^2 * V(ϕ) - dϕ^2) / 6.0
    # (a''/a)'
    app_a_p = diff(app_a[1:end-1]) ./ diff(η[1:end-1])
    Ω_r = @. ρ_r/(3*H^2)
    Ω_ϕ = @. ρ_ϕ/(3*H^2)

    # a_e, H_e = get_end(sol)
    # @info log(a_e) interpolate(a, ϕ, a_e) H_e
    H_e = interpolate(a, H, a_e)
    # @show a_e, H_e, log(a_e)

    dec_index = findfirst(x -> x <= Γ, H)
    # a_rh = a[dec_index]
    a_rh = try
        # reheating scale factor
        a[dec_index]
    catch e
        @warn "Scalar factor at reheating not found! %f"
        a[end]
    end
    
    t = cumul_integrate(η, a)
    w = @. get_eos(ϕ, dϕ, a, V, ρ_r, α)
    V = @. V(ϕ)

    # now need to discard the last two elements
    # as app_a_p miss these
    return EOMData(η[1:end-2], ϕ[1:end-2], dϕ[1:end-2], 
            a[1:end-2], app_a[1:end-2], app_a_p,
            H[1:end-2], Ω_r[1:end-2], Ω_ϕ[1:end-2],
            a_e, a_rh, H_e, t[1:end-2], w[1:end-2], V[1:end-2])
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
    
    t::V
    w::V
    V::V
end

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
    "H_e" => eom.Hₑ,
    
    "t" => eom.t,
    "w" => eom.w,
    "V" => eom.V
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

#=
"""
methods with additional argument: m_eff 
mainly for TModel
"""
function save_eom_npz(eom::EOMData, data_dir, m_eff)
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
    "H_e" => eom.Hₑ,
    
    "t" => eom.t,
    "w" => eom.w,
    "V" => eom.V,

    "m_eff" => m_eff.(eom.ϕ)
    ))
end


function save_all(u₀, tspan, p, data_dir, m_eff)
    eom_data = @time solve_eom(u₀, tspan, p)

    mkpath(data_dir)
    # serialize for Bogoliubov computation
    serialize(data_dir * "eom.dat", eom_data)
    save_eom_npz(eom_data, data_dir, m_eff)
    return nothing
end
=#
end
