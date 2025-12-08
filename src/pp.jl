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
    
    condition(u, t, integrator) = (abs(1-get_ω2(t)/k^2) < 1e-4)
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)
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

#######################################################################
# Analytical Bogobiubov 
#######################################################################

"""
get the Bogoliubov coefficient analytically
"""
# TODO: implement the phase
function get_beta_bogo_ana(eom, k::Vector, model::Symbol, dn)
    # how many fourier modes to compute
    # for n=2, it doesn't need to be very large
    if model == :quadratic
        num_j = 10
    else 
        num_j = 20 
    end

    #=
    Process the eom arrays first
    apply mask to focus after end of inflation

    interpolation does nothing, don;t want to change now
    =#
    # a_mask = eom.a .>= eom.aₑ/5
    a_mask = eom.a .>= eom.aₑ
    # t_new, V_new = @time _get_dense(eom.t[a_mask], eom.V[a_mask])
    t_new = eom.t[a_mask]
    V_new = eom.V[a_mask]
    H_new = Interpolate(eom.t[a_mask], eom.H[a_mask]).(t_new)
    a_new = Interpolate(eom.t[a_mask], eom.a[a_mask]).(t_new)
    ρ_ϕ = @. eom.Ω_ϕ * 3 * eom.H^2
    ρ_new = Interpolate(eom.t[a_mask], ρ_ϕ[a_mask]).(t_new)
    # @show a_new[1], H_new[1]
    # @info "a_e H_e / a H =" eom.aₑ * H_new[1] ./ (H_new .* a_new) 
    
    N, indices, m_tilde, c_n, mdm2 = get_four_coeff(num_j, t_new, V_new ./ ρ_new, dn)
    # @show size(m_tilde), size(indices)
    # @info @sprintf "At first minimum: a/a_e = %e" a_new[indices[1]]/a_new[1]
    # @info @sprintf "At first minimum: m_tilde/H = %e" m_tilde[1] / H_new[1]
    @info @sprintf "At first minimum: k/a_e H_e = %e" a_new[indices[1]]/a_new[1] * m_tilde[1] / H_new[1]
    
    dm = get_deriv_BSpline(t_new[indices[1:end-1]], m_tilde, 2, 1)
    ddm = get_deriv_BSpline(t_new[indices[1:end-1]], m_tilde, 3, 2)
    # display(log.(abs.(ddm)))
    # @show N, size(indices), size(dm), size(ddm)
    # size of dm, ddm: N (# of oscillations)
    # size of indices: N+1
    
    f = zeros(size(k))
    for j in 1:num_j
    # interate over Fourier modes
        X = @. a_new[indices[1:end-1]] * (dm*t_new[indices[1:end-1]] + m_tilde) * j / a_new[1] / H_new[1]
        # X = @. a_new[indices[1:end-1]] * (2 / 6 * m_tilde) * j / a_new[1] / H_new[1]
        # @show size(X)

        # meant to be the (magnitude of) Bogo. coefficient
        Y = zeros(N)
        Threads.@threads for i in 1:N
            i2 = indices[i]
            # |g''|
            gpp = abs(2*j*a_new[i2]^2 * (ddm[i]*t_new[i2] + dm[i]*(2+t_new[i2]*H_new[i2]) + m_tilde[i]*H_new[i2]))
            # use "quasi"-analytical expression
            # gpp = 2*j * (a_new[i2] * H_new[i2])^2 * m_tilde[i] / H_new[i2]

            Y[i] = 3/2 * (a_new[i2] * H_new[i2])^3 / (a_new[1] * H_new[1])^2 / X[i]^2 * (dm[i]*t_new[i2] + m_tilde[i])/H_new[i2] * abs(c_n[i, j]) * sqrt(π/gpp) 
        end
        # @show size(X) size(Y)
        # @show X[1]

        if j == 1
            display(X)
            display(Y)
        end

        if model == :quadratic
            tmp = Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += abs2.(tmp)
        elseif model == :quartic 
            tmp = Interpolate(X[1:argmax(X)], Y[1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += abs2.(tmp)
            f += Interpolate(X[end:-1:argmax(X)], Y[end:-1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        else
            f += Interpolate(reverse(X), reverse(Y), extrapolate=LinearInterpolations.Constant(0.0)).(k)
        end
    end
    # @show k, β
    return f
end

function save_all_ana(num_k, data_dir, model, log_k_i=0, log_k_f=2)
    @info data_dir
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log_k_i, log_k_f, num_k)
    f = @time get_beta_bogo_ana(eom, k, model, data_dir)

    npzwrite(data_dir * "spec_bogo_ana.npz", Dict(
        "k" => k,
        "f" => f,
    ))
end

end
