module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf, Serialization
#  using JLD2
#  using Infiltritor
 using Interpolations

using ..Commons

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

function save_all(num_k, data_dir)
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log10(2), log10(500), num_k) * eom.aₑ * eom.Hₑ
    # k = @. [1.5] * a_e * H_e
    
    n, ρ, error = @time PPs.get_f(k, eom)
    # @show n, ρ, error
    
    mkpath(data_dir)
    npzwrite(data_dir * "spec.npz", Dict(
        "k" => k ./ (eom.aₑ*eom.Hₑ),
        "n" => abs.(n),
        "rho" => abs.(ρ ./ (eom.aₑ*eom.Hₑ)^4),
        "error" => error
    ))
end

end
