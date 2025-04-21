module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf, Serialization
#  using JLD2
#  using Infiltritor
 using Interpolations

using ..Commons

#=
"""
Defines the differential equation to solve
"""
function get_diff_eq(u::SVector, p::Tuple, t::Real)
    ω = p[1]
    dω = p[2]
    Ω = p[3]
    # @show t, ω(t), dω(t)

    α = u[1]
    β = u[2]

    dω_2ω = dω(t) / (2 * ω(t))
    e = exp(+2.0im * Ω(t))
    dydt = dω_2ω .* SA[e * β, conj(e) * α]
    return dydt
end

function get_alpha_beta_domain(u, p, t)
    if (abs2(u[1]) - abs2(u[2]) - 1) < 1e-12
        return false
    else 
        return true
    end
end

"""
get the parameter for ODESolver ready;
They are interpolator of ω, ω', Ω
"""
function get_p(k::Real, t, app_a, app_a_p)
    # ω^2 = k^2 - a''/a
    ω = @. sqrt(Complex(k^2 - app_a))
    dω = @. - app_a_p / (2.0 * ω)
    Ω = cumul_integrate(t, ω)

    get_ω = LinearInterpolations.Interpolate(t, ω)
    get_dω = LinearInterpolations.Interpolate(t, dω)
    get_Ω = LinearInterpolations.Interpolate(t, Ω)

    # Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    
    return get_ω, get_dω, get_Ω
end

function get_f(k::Vector, eom) 
    if check_array(eom.app_a) | check_array(eom.app_a_p)
        throw(ArgumentError("Input arrays contain NaN or Inf!"))
    end

    # @show k[1], k[end]
    # @show _app(0.0) / _a(0.0)

    α = zeros(ComplexF64, size(k)) 
    β = zeros(ComplexF64, size(k)) 
    ω = zeros(ComplexF64, size(k)) 

    Threads.@threads for i in eachindex(k)
        p = get_p(k[i], eom.τ, eom.app_a, eom.app_a_p)
        
        tspan = [eom.τ[1], eom.τ[end]]
        u₀ = SA[1.0 + 0.0im, 0.0+0.0im]

        prob = ODEProblem{false}(get_diff_eq, u₀, tspan, p)
        # sol = @time solve(prob, RK4())
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, save_everystep=false, maxiters=1e8, isoutofdomain=get_alpha_beta_domain)

        α[i] = sol[1, end]
        β[i] = sol[2, end]
        ω[i] = p[1](sol.t[end])
    end
    # @show α, β, ω
    # @show abs2.(β)
    n = abs2.(β)
    ρ = @. ω * abs2(β) * k^3 / π^2
    error = @. abs2(α) - abs2(β) - 1
    return n, ρ, error
end
=#

function get_diff_eq_mode(u, ω2, t)
    # @show t, real(1.0+1.0im * get_wronskian(u[1], u[2]))
    χ = u[1]
    ∂χ = u[2]

    return @SVector [∂χ, -ω2(t)*χ]
end

"""
should be i
"""
function get_wronskian(χ, ∂χ)
    return χ * conj(∂χ) - conj(χ) * ∂χ
end

function get_wronskian_domain(u, p, t)
    # assume get_wronskian returns a pure img. number
    if real(1.0+1.0im * get_wronskian(u[1], u[2])) < 1e-5
        return false
    else 
        return true
    end
end

"""
f = |β|^2 from mode functions χₖ, ∂χₖ
"""
function _get_f(ω, χ, ∂χ)
    return abs2(ω * χ - 1.0im * ∂χ)/(2*ω)
end

function solve_diff_mode(k::Real, eom)
    tspan = [eom.τ[1], eom.τ[end]]

    # ω^2 = k^2 - a''/a
    get_app_a = LinearInterpolations.Interpolate(eom.τ, eom.app_a)
    get_ω2 = x -> k^2 - get_app_a(x)
    ω₀ = sqrt(get_ω2(tspan[1])+0.0im)
    ωₑ = sqrt(get_ω2(tspan[2])+0.0im)

    # u₀ = @SVector [1/sqrt(2*k), -1.0im*k/sqrt(2*k)] 
    u₀ = @SVector [1/sqrt(2*ω₀), -1.0im*ω₀/sqrt(2*ω₀)] 
    
    prob = ODEProblem{false}(get_diff_eq_mode, u₀, tspan, get_ω2)
    sol = solve(prob, Vern9(), reltol=1e-10, abstol=1e-10, save_everystep=false, isoutofdomain=get_wronskian_domain)
    χₑ = sol[1, end]
    ∂χₑ = sol[2, end]

    # wronskian
    err = 1 + 1.0im * get_wronskian(χₑ, ∂χₑ)

    n = _get_f(ωₑ, χₑ, ∂χₑ)
    ρ = n * ωₑ* k^3 / π^2
    return n, ρ, err
end

function solve_all_spec(k::Vector, eom)
    if check_array(eom.app_a) | check_array(eom.app_a_p)
        throw(ArgumentError("Input arrays contain NaN or Inf!"))
    end

    n = zeros(size(k))
    ρ = zeros(size(k))
    err = zeros(size(k))

    @inbounds Threads.@threads for i in eachindex(k)
        @inbounds res = solve_diff_mode(k[i], eom)
        @inbounds n[i], ρ[i], err[i] = res
        # @show res
    end
    
    # @show n, ρ, error
    return n, ρ, err
end

function save_all(num_k, data_dir)
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log10(2), log10(500), num_k) * eom.aₑ * eom.Hₑ
    # k = @. [1.5] * a_e * H_e
    
    n, ρ, err = @time solve_all_spec(k, eom)
    # @show n, ρ, err
    
    mkpath(data_dir)
    npzwrite(data_dir * "spec.npz", Dict(
        "k" => k ./ (eom.aₑ*eom.Hₑ),
        "n" => abs.(n),
        "rho" => abs.(ρ ./ (eom.aₑ*eom.Hₑ)^4),
        "error" => err
    ))
end

end
