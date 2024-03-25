module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf
#  using JLD2
#  using Infiltritor
#  using Interpolations

using ..Commons

# type of interpolator...
const INTERPOLATOR_TYPE = LinearInterpolations.Interpolate{typeof(LinearInterpolations.combine), Tuple{Vector{Float64}}, Vector{Float64}, Symbol}
#  const INTERPOLATOR_TYPE = Interpolations.GriddedInterpolation{Float64, 1, Vector{Float64}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{Float64}}}

"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle

One is responsible to make sure ode.τ and m2_eff has the same dimension. Now ode data isshould be automatically trimmed.

LinearInterpolations uses ~ half as much as memories, a bit faster also.
"""
function init_Ω(k::Real, τ::Vector, m2::Vector)
    # get sampled ω values and interpolate
    ω = @. (k^2 + m2)^(1/2)
    # cumulative integration
    Ω = cumul_integrate(τ, ω)
    get_Ω = LinearInterpolations.Interpolate(τ, Ω)
    #  get_Ω = Interpolations.interpolate((τ,), Ω, Gridded(Linear()))

    return get_Ω
end

"""
Defines the differential equation to solve
"""
function get_diff_eq(u::SVector, p::Tuple, t::Real)
    ω = p[1]
    dω = p[2]
    Ω = p[3]

    α = u[1]
    β = u[2]

    dω_2ω = dω(t) / (2 * ω(t))
    e = exp(+2.0im * Ω(t))
    dydt = dω_2ω .* SA[e * β, conj(e) * α]
    return dydt
end

"""
get the parameters (interpolators) for diff_eq
"""
function get_p(k::Real, τ::Vector, m2::Vector, get_m2::INTERPOLATOR_TYPE, get_dm2::INTERPOLATOR_TYPE)
    ω = x -> sqrt(k^2 + get_m2(x))
    dω = x -> get_dm2(x) / (2*ω(x))
    #  @show typeof(ω) typeof(dω) typeof(Ω)
    return ω, dω, init_Ω(k, τ, m2)
end

"""
Solve the differential equations for GPP
dtmax not used, but keep just in case
NOTE: max_err is now deprecated!
"""
function solve_diff(k::Real, τ::Vector, m2::Vector, get_m2::INTERPOLATOR_TYPE, get_dm2::INTERPOLATOR_TYPE)
    p = get_p(k, τ, m2, get_m2, get_dm2)
    t_span = [τ[1], τ[end-2]]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]
    
    # false: out of place function for ODEs
    prob = ODEProblem{false}(get_diff_eq, u₀, t_span, p)
    #  adaptive algorithm depends on relative tolerance
    sol = solve(prob, RK4(), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e8)

    #  res = sol.u[end]
    #  αₑ = sol[1, end]
    βₑ = sol[2, end]
    f = abs(βₑ)^2
    #  max_err = abs(abs(αₑ)^2 - abs(βₑ)^2 - 1)

    #  return f, max_err
    return f
end 

function solve_diff(k::Vector, τ::Vector, m2::Vector, get_m2::INTERPOLATOR_TYPE, get_dm2::INTERPOLATOR_TYPE)
    f = zeros(size(k)) 
    #  err = zeros(size(k))
    
    Threads.@threads for i in eachindex(k)
        @inbounds res = solve_diff(k[i], τ, m2, get_m2, get_dm2)
        #  @show typeof(res), res
        @inbounds f[i] = res
        #  @inbounds err[i] = res[2]
    end
    
    #  return f, err
    return f
end

###########################################################################

function solve_diff_ensemble(k::Vector, τ::Vector, m2::Vector, get_m2::INTERPOLATOR_TYPE, get_dm2::INTERPOLATOR_TYPE)
    t_span = [τ[1], τ[end-2]]
    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]
    parameters = [get_p(x, τ, m2, get_m2, get_dm2) for x in k]

    prob = ODEProblem(get_diff_eq, u₀, t_span, parameters[1])

    #  function prob_func(prob, i, repeat)
        #  remake(prob, p = parameters[i])
    #  end
    #  let: ensure type stability
    prob_func = let parameters = parameters
        (prob, i, repeat) -> begin
            remake(prob, p = parameters[i])
        end 
    end

    function output_func(sol, i)
        #  α = sol[1, end]
        β = sol[2, end]
        f = abs(β)^2
        #  err = abs(abs(α)^2 - abs(β)^2 - 1)
        return f, false
    end
    
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, output_func=output_func)
    sim = solve(ensemble_prob, RK4(), EnsembleThreads(), trajectories = length(k), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e8)
    # dump(sim)
    return sim[:]
end

function test_ensemble(k::Vector, ode::ODEData, get_m2_eff::Function, mᵪ::Real, ξ::Real)
    m2_eff = get_m2_eff(ode, mᵪ, ξ)
    get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff)
    get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ))

    @show solve_diff_ensemble(k, ode.τ, m2_eff, get_m2, get_dm2)[:, 1]
    #  @time solve_diff(k, ode.τ, m2_eff, get_m2, get_dm2)
    return true
end

###########################################################################
#
"""
compute comoving energy (a⁴ρ) given k, m2, and f arrays
"""
function get_com_energy(k::Vector, f::Vector, m2::Real)
    ω = sqrt.(@. k^2 + m2)
    integrand = @. k^2 * ω * f / (4*π^2)
    return integrate(k, integrand)
end

function get_com_number(k::Vector, f::Vector)
    integrand = @. k^2 * f / (4*π^2) 
    return integrate(k, integrand)
end

"""
save the spectra for various parameters (given as arguments);
use multi-threading, remember use e.g. julia -n auto
direct_out -> if return the results instead of save to npz

results data structure:
- ModelName
    - f_ξ=ξ
        m_χ=m_χ.npz

"""
function save_each(data_dir::String, mᵩ::Real, ode::ODEData, 
                   k::Vector, mᵪ::SVector, ξ::SVector, 
                   get_m2_eff::Function;
                   direct_out::Bool=false,
                   fn_suffix::String="")
    # interate over the model parameters
    for ξᵢ in ξ
        ρs = zeros(MVector{size(mᵪ)[1]})
        ns = zeros(MVector{size(mᵪ)[1]})
        f0s = zeros(MVector{size(mᵪ)[1]})
        ξ_dirᵢ = data_dir * "f_ξ=$ξᵢ/"
        
        iter = ProgressBar(eachindex(mᵪ))
        for i in iter
            @inbounds mᵪᵢ = mᵪ[i]
            set_description(iter, ("mᵪ: $(@sprintf("%.2f", mᵪᵢ))"))
            #  only want to compute this once for one set of parameters
            m2_eff = get_m2_eff(ode, mᵪᵢ, ξᵢ)
            get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff)
            #  get_m2 = Interpolations.interpolate((ode.τ,), m2_eff, Gridded(Linear()))
            # dm2 = d(m^2)/dτ
            get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ))
            #  get_dm2 = Interpolations.interpolate((ode.τ[1:end-1],), diff(m2_eff) ./ diff(ode.τ), Gridded(Linear()))
            
            f = solve_diff(k, ode.τ, m2_eff, get_m2, get_dm2)
            #  f = solve_diff_ensemble(k, ode.τ, m2_eff, get_m2, get_dm2)
            
            # take the ρ at the end, use last m2_eff
            @inbounds ρs[i] = get_com_energy(k, f, m2_eff[end-2])
            @inbounds ns[i] = get_com_number(k, f)
            @inbounds f0s[i] = f[1]

            if direct_out
                return f
            else
                mkpath(ξ_dirᵢ)
                npzwrite("$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix.npz",
                         Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f
                              #  , "err"=>err
                             ))
            end
        end
        # k is in planck unit
        # want ρ and n in planck unit as well
        # add all other factors in the plotting
        #  @show ξ_dirᵢ fn_suffix
        #  @show typeof(mᵪ / mᵩ) typeof(f0s) typeof(ρs) typeof(ns)
        npzwrite("$(ξ_dirᵢ)integrated$fn_suffix.npz",
                 Dict("m_chi" => [mᵪ / mᵩ ...], "f0"=>[f0s...], "rho"=>[ρs...], "n"=>[ns...]))
    end
end

"""
adding one more iteration over m3_2
results data structure:
- ModelName
    - m3_2=m3_2
        - f_ξ=ξ
            m_χ=m_χ.npz
"""
function save_each(data_dir::String, mᵩ::Real, ode::ODEData, 
                   k::Vector, mᵪ::SVector, ξ::SVector, 
                   m3_2::SVector,
                   get_m2_eff::Function;
                   direct_out::Bool=false,
                   fn_suffix::String="")
    iter = ProgressBar(eachindex(m3_2))
    for i in iter
        @inbounds m = m3_2[i]
        set_description(iter, "m_32: $(@sprintf("%.2f", m))",)
        m3_2_dir = data_dir * "m3_2=$(m/mᵩ)/"
        m2_eff_R(ode, mᵪ, ξ) = get_m2_eff(ode, mᵪ, ξ, m)
        save_each(m3_2_dir, mᵩ, ode, k, mᵪ, ξ, m2_eff_R, 
                  direct_out=direct_out, fn_suffix=fn_suffix)
    end
end

end
