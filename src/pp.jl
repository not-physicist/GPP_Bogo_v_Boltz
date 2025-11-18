module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf, Serialization
#  using JLD2
#  using Infiltritor
using Interpolations

using ..Commons

"""
Defines the differential equation to solve
"""
function get_diff_eq_alpha(u, p, t)
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
function get_p_alpha(k::Real, t, app_a, app_a_p)
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

function solve_all_spec_alpha(k::Vector, eom) 
    if check_array(eom.app_a) | check_array(eom.app_a_p)
        throw(ArgumentError("Input arrays contain NaN or Inf!"))
    end

    # @show k[1], k[end]
    # @show _app(0.0) / _a(0.0)

    α = zeros(ComplexF64, size(k)) 
    β = zeros(ComplexF64, size(k)) 
    ω = zeros(ComplexF64, size(k)) 

    Threads.@threads for i in ProgressBar(eachindex(k))
        p = get_p_alpha(k[i], eom.τ, eom.app_a, eom.app_a_p)
        
        tspan = [eom.τ[1], eom.τ[end]]
        u₀ = SA[1.0 + 0.0im, 0.0+0.0im]

        prob = ODEProblem{false}(get_diff_eq_alpha, u₀, tspan, p)
        # sol = @time solve(prob, RK4())
        sol = solve(prob, Vern9(), reltol=1e-12, abstol=1e-12, save_everystep=false, maxiters=1e8, isoutofdomain=get_alpha_beta_domain)
        # sol = solve(prob, AutoVern9(Rodas5P(autodiff=false)), reltol=1e-12, abstol=1e-12, save_everystep=false, maxiters=1e8, isoutofdomain=get_alpha_beta_domain)

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

#######################################################################################################################
#######################################################################################################################

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
    sol = solve(prob, Vern9(), reltol=1e-12, abstol=1e-12, save_everystep=false, isoutofdomain=get_wronskian_domain, maxiters=1e7)
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

    @inbounds Threads.@threads for i in ProgressBar(eachindex(k))
        @inbounds res = solve_diff_mode(k[i], eom)
        @inbounds n[i], ρ[i], err[i] = res
        # @show res
    end
    
    # @show n, ρ, error
    return n, ρ, err
end

function save_all(num_k, data_dir, log_k_i = 0, log_k_f = 2)
    @info data_dir
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log_k_i, log_k_f, num_k) * eom.aₑ * eom.Hₑ

    # "critical" comoving momenta: largest k with tachyonic instab.
    # in mpl unit
    k_c = 2*sqrt(maximum(eom.app_a))
    @info "k_c/a_e H_e = " k_c/(eom.aₑ * eom.Hₑ)
    
    n1, ρ1, err1 = @time solve_all_spec(k[k .<= k_c], eom)
    n2, ρ2, err2 = @time solve_all_spec_alpha(k[k .> k_c], eom)
    n = [n1; n2]
    ρ = [ρ1; ρ2]
    err = [err1; err2]
    # @info size(n) size(ρ) size(err)
    # @info log.(n)
    # @info size(f_boltz)
    
    # mkpath(data_dir)
    npzwrite(data_dir * "spec_bogo.npz", Dict(
        "k" => k ./ (eom.aₑ*eom.Hₑ),
        "n" => abs.(n),
        "rho" => abs.(ρ ./ (eom.aₑ*eom.Hₑ)^4),
        "error" => err
    ))
    return nothing
end

"""
save at every step
"""
function solve_diff_mode_every(k::Real, eom)
    tspan = [eom.τ[1], eom.τ[end]]

    # ω^2 = k^2 - a''/a
    get_app_a = LinearInterpolations.Interpolate(eom.τ, eom.app_a)
    get_ω2 = x -> k^2 - get_app_a(x)
    ω₀ = sqrt(get_ω2(tspan[1])+0.0im)
    ωₑ = sqrt(get_ω2(tspan[2])+0.0im)

    # u₀ = @SVector [1/sqrt(2*k), -1.0im*k/sqrt(2*k)] 
    u₀ = @SVector [1/sqrt(2*ω₀), -1.0im*ω₀/sqrt(2*ω₀)] 
    
    prob = ODEProblem{false}(get_diff_eq_mode, u₀, tspan, get_ω2)
    sol = solve(prob, Vern9(), reltol=1e-12, abstol=1e-12, save_everystep=true, isoutofdomain=get_wronskian_domain, maxiters=1e7)

    χ = sol[1, :]
    ∂χ = sol[2, :]

    # wronskian
    err = @. 1 + 1.0im * get_wronskian(χ, ∂χ)

    n = @. 2 * _get_f(sqrt(get_ω2(sol.t) + 0.0im), χ, ∂χ)
    # @info size(eom.τ) size(eom.a)
    get_a = LinearInterpolations.Interpolate(eom.τ, eom.a)
    # @info size(sol.t)
    N = @. log(get_a(sol.t))
    # @info size(N)
    return N, n, err
end

"""
saving the time evolution of |β|^2
"""
function save_all_every(data_dir)
    @info data_dir 
    mkpath(data_dir)
    eom = deserialize(data_dir * "eom.dat")

    num_k = 10 
    k = @. logspace(-1, 1, num_k) * eom.aₑ * eom.Hₑ 
    for ki in k
        N, n, err = solve_diff_mode_every(ki, eom)
        fn = @sprintf "%sk=%.1e.npz" data_dir ki/(eom.aₑ*eom.Hₑ)
        # @info k fn 
        npzwrite(fn, Dict(
        "N" => N,
        "n" => abs.(n),
        "error" => err
        ))
    end
end

end
