"""
module to compute exact Boltzmann result
"""
module Boltzmann


# using CubicSplines: extrapolate
using ..Commons

using LinearInterpolations, NumericalIntegration, Peaks, Statistics, Serialization, NPZ, QuadGK, ProgressBars
# using BSplineKit 
using CurveFit
# TODO: clean up unused package

"""
compute the Boltzmann phase space distribution
decompose the oscillations of inflaton into fourier series

k in unit of aₑHₑ
"""
function get_f(eom, k::Vector, model::Symbol, dn, single=false)
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
    
    N, indices, m_tilde, c_n, mdm2 = Commons.get_four_coeff(num_j, t_new, V_new ./ ρ_new, dn)
    # @show size(indices) size(ωj) size(ωpω2)
    # @show mdm2
    
    #=
    # do a curve fir of ω(a) of first 10 elements
    @info "a * ωj = " a_new[indices[1:end-1]] .* ωj[:, 1]
    fitf = curve_fit(PowerFit, a_new[indices[1:5]], ωj[1:5, 1])
    @show fitf
    @show ωj[:, 1]
    @show fitf.(a_new[indices])
    =#
    
    #=
    Now compute the spectrum
    First compute the f only at the maxima (given by indices)
    Then interpolate for dense output
    =#
           
    # First method: good for n=2, n=4 a bit strange
    f = zeros(size(k))
    for j in 1:num_j
        # iterate over j (different fourier modes)
        n2_H = zeros(N)
        Threads.@threads for i in 1:N
            n2_H[i] = π/4 / (m_tilde[i]*j)^3 * ρ_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]]
        end
        
        # to be interpolated as k/a_e H_e
        X = @. a_new[indices[1:end-1]] * m_tilde * j / a_new[1] / H_new[1]
        # @info "Production of first inflaton oscillation at k/a_e H_e = " X[1]
        # @show X

        # correction factor 
        C = @. 1/abs(1 + 1/j * X * eom.aₑ * H_new[1] / (a_new * H_new)[indices[1:end-1]] * (mdm2))
        
        #=
        if j == 1
            display(X)
            # display(n2_H)
            # @info "Correction factor:"
            # display(C)
        end
        =#

        Y = n2_H .* C

        if single && j <= 5
            fn = dn * "spec_boltz_j=$j.npz"
            npzwrite(fn, Dict("X" => X, "Y" => Y))
        end

        # find the UV cutoff (for n > 4)

        if model == :quadratic
            n2_H_dense = Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
        elseif model == :quartic
            n2_H_dense = Interpolate(X[1:argmax(X)], Y[1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
            f += Interpolate(X[end:-1:argmax(X)], Y[end:-1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        elseif model == :sextic
            # sort_ind = sortperm(X[1:end-1])
            f += Interpolate(reverse(X), reverse(Y), extrapolate=LinearInterpolations.Constant(0.0)).(k)
        else
            return ArgumentError("Model not implemented!")
        end
    end
    return f, a_new[indices], m_tilde
    
    #=
    # To get uninterpolated spectrum
    # Third method: n=2 spotty, but not so bad, bad for n=4
    k_new = Vector{Float64}(undef, 0)
    f = Vector{Float64}(undef, 0)
    for i in 1:N
    # iterate over different oscillations
        X = a_new[indices[i]] .* ωj[i, :] ./ 2 / eom.aₑ / eom.Hₑ
        # ~ n^2 / H
        Y = [4*π/ωj[i, j]^3 * V_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]] for j in 1:num_j]
        k_new = [k_new; X]
        f = [f; Y]
    end
    sort_ind = sortperm(k_new)
    @show k_new[sort_ind], f[sort_ind]
    # return Interpolate(k_new[sort_ind], f[sort_ind], extrapolate=LinearInterpolations.Constant(0.0)).(k)
    return k_new[sort_ind], f[sort_ind]
    =#
end

function save_all(num_k, data_dir, model, log_k_i=0, log_k_f=2, single=false)
    @info data_dir
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log_k_i, log_k_f, num_k)
    f, a, m = @time get_f(eom, k, model, data_dir, single)
    
    # display(k)
    # display(f)
    npzwrite(data_dir * "spec_boltz.npz", Dict(
        "k" => k,
        "f" => f,
    ))

    npzwrite(data_dir * "m_tilde.npz", Dict(
        "a" => a,
        "m" => m,
    ))

    return nothing
end

end
