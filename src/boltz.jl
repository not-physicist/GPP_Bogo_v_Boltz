"""
module to compute exact Boltzmann result
"""
module Boltzmann

using CubicSplines: extrapolate
using ..Commons

using CubicSplines, LinearInterpolations, NumericalIntegration, Peaks, Statistics, Serialization, NPZ, QuadGK

"""
interpolate "signal" using cubic spline
"""
function _get_dense(x, y)
    Δx = minimum(diff(x)) / 5
    x_new = range(x[1], x[end], step=Δx)
    intp = CubicSpline(x, y)
    y_new = intp[x_new]
    return x_new, y_new 
end

"""
get (complex) a single fourier coefficient using the integral formula
assume the input is exactly one period
"""
function _get_c_n(x, y, P, n::Int)
    ω = 2*π*n/P
    
    y_itp = Interpolate(x, y)
    res = quadgk(k -> y_itp(k)*exp(-1.0im*ω*k), x[1], x[end])[1]
    cₙ = 1/P * res[1]
    return ω, cₙ
end

"""
    Get Fourier series for each oscillation
    separate arrays for frequencies and four. coeffcients 
    two-dimensional: time and multiples of fundamental frequency
"""
function get_four_coeff(num_j, t, V_ρ)
    indices = @time argmaxima(V_ρ)
    # N = size(indices)[1] - 1
    # @show N
    # @show indices
    
    N = size(indices)[1]
    pushfirst!(indices, 1)

    ωj = zeros(Float64, (N, num_j))
    c_n = zeros(ComplexF64, (N, num_j))
    Threads.@threads for i in 1:N 
        i1 = indices[i]
        i2 = indices[i+1]
    
        # get t and (normalized) V for a single oscillation
        t_osci = t[i1:i2] .- t[i1]
        V_osci = (V_ρ)[i1:i2]
        P = t_osci[end] - t_osci[1]
        tmp = [_get_c_n(t_osci, V_osci, P, j) for j in 1:num_j]
        ωj[i, :] = [x[1] for x in tmp]
        c_n[i, :] = [x[2] for x in tmp]
    end
    return N, indices, ωj, c_n
end

"""
compute the Boltzmann phase space distribution
decompose the oscillations of inflaton into fourier series

k in unit of aₑHₑ
"""
function get_f(eom, k::Vector, model::Symbol)
    # how many fourier modes to compute
    # for n=2, it doesn't need to be very large
    if model == :quadratic
        num_j = 10
    else 
        num_j = 100 
    end

    #=
    Process the eom arrays first
    apply mask to focus after end of inflation
    interpolate the input, better fourier series
    =#
    # @show eom.t[1], (eom.t[eom.a .>= eom.aₑ])[1]*1e-5/π
    a_mask = eom.a .>= eom.aₑ
    # t_new, V_new = @time _get_dense(eom.t[a_mask], eom.V[a_mask])
    t_new = eom.t[a_mask]
    V_new = eom.V[a_mask]
    H_new = Interpolate(eom.t[a_mask], eom.H[a_mask]).(t_new)
    a_new = Interpolate(eom.t[a_mask], eom.a[a_mask]).(t_new)
    ρ_ϕ = @. eom.Ω_ϕ * 3 * eom.H^2
    ρ_new = Interpolate(eom.t[a_mask], ρ_ϕ[a_mask]).(t_new)
    
    N, indices, ωj, c_n = get_four_coeff(num_j, t_new, V_new ./ ρ_new)
    # show (j=1) ω for different times
    @show ωj[1:end, 1]
    @show abs.(c_n[1, :]), abs.(c_n[end, :])
    
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
            n2_H[i] = 4*π/ωj[i, j]^3 * V_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]]
        end
        
        # to be interpolated as k/a_e H_e
        X = a_new[indices[1:end-1]] .* ωj[:, j] ./ 2 / eom.aₑ / eom.Hₑ
        if j == 1
            @show X
        end
        # indices for the sorted array
        Y = n2_H
        # @info X[1:argmax(X)] X[argmax(X):end]
        # @info X[1]
        if model == :quadratic
            n2_H_dense = Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
        elseif model == :quartic
            n2_H_dense = Interpolate(X[1:argmax(X)], Y[1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
            f += Interpolate(X[end:-1:argmax(X)], Y[end:-1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        elseif model == :sextic
            sort_ind = sortperm(X)
            f += Interpolate(X[sort_ind], Y[sort_ind], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        else
            return ArgumentError("Model not implemented!")
        end
    end
    return f
    
    #=
    # Second method: n=2 spectrum enhanced for whatever reason, n=4 alright
    f = zeros(size(k))
    for i in 1:N
    # iterate over different oscillations
        X = a_new[indices[i]] .* ωj[i, :] ./ 2 / eom.aₑ / eom.Hₑ
        # @show X
        # ~ n^2 / H
        Y = [4*π/ωj[i, j]^3 * V_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]] for j in 1:num_j]
        f += Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
    end
    =#
    
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

function save_all(num_k, data_dir, model, log_k_i=0, log_k_f=2)
    @info data_dir
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log_k_i, log_k_f, num_k)
    f = @time get_f(eom, k, model)

    # mkpath(data_dir)
    npzwrite(data_dir * "spec_boltz.npz", Dict(
        "k" => k,
        "f" => f,
    ))
    return nothing
end
end
