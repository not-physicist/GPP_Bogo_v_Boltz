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
    Δx = minimum(diff(x)) / 2
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
    # x_new = range(x[1], x[end], step=minimum(diff(x))/(20*n))
    # y_new = Interpolate(x, y).(x_new)

    # integrand = @. y_new*exp(-1.0im*ω*x_new)
    # @show integrand
    # cₙ = 1/P * integrate(x_new, integrand)
    
    y_itp = Interpolate(x, y)
    res = quadgk(k -> y_itp(k)*exp(-1.0im*ω*k), x[1], x[end])[1]
    cₙ = 1/P * res[1]
    return ω, cₙ
end

"""
compute the Boltzmann phase space distribution
decompose the oscillations of inflaton into fourier series

k in unit of aₑHₑ
"""
function get_f(eom, k::Vector)
    num_j = 100

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
    
    #=
    Find maxima of V(t)
    =#
    indices = @time argmaxima(V_new)
    # add first "oscillation" gives lower momenta, but UV end will be messed up
    # pushfirst!(indices, 1)
    periods = diff(t_new[indices])
    # @show diff(log10.(periods))
    # @show V_new[indices]
    # @show diff(log.(periods)) ./ diff(log.(t_new[indices[1:end-1]]))
    # @show mean(periods), std(periods)
    N = size(indices)[1] - 1
    # @info V_new[1], V_new[indices[1]]
    # @info @. a_new[indices[1:end-1]] * (2π/periods) / (2 * eom.aₑ * eom.Hₑ)
    
    #=
    Get Fourier series for each oscillation
    separate arrays for frequencies and four. coeffcients 
    two-dimensional: time and multiples of fundamental frequency
    =#
    ωj = zeros(Float64, (N, num_j))
    c_n = zeros(ComplexF64, (N, num_j))
    Threads.@threads for i in 1:N 
        i1 = indices[i]
        i2 = indices[i+1]
        # @show P
    
        # get t and V for a single oscillation
        t_osci = t_new[i1:i2] .- t_new[i1]
        V_osci = (V_new ./ ρ_new)[i1:i2]
        P = t_osci[end] - t_osci[1]
        tmp = [_get_c_n(t_osci, V_osci, P, j) for j in 1:num_j]
        ωj[i, :] = [x[1] for x in tmp]
        c_n[i, :] = [x[2] for x in tmp]
    end
    # show (j=1) ω for different times
    # @show ωj[1:end, 1]
    @show abs.(c_n[1, :]), abs.(c_n[end, :])
    
    # @show size(ωj), size(t_new[indices])

    #=
    Now compute the spectrum
    First compute the f only at the maxima (given by indices)
    Then interpolate for dense output
    =#


    #=
    # First method: godd for n=2, n=4 a bit strange
    f = zeros(size(k))
    for j in 1:num_j
        # iterate over j (different fourier modes)
        n2_H = zeros(N)
        Threads.@threads for i in 1:N
            n2_H[i] = 4*π/ωj[i, j]^3 * V_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]]
        end
        
        X = a_new[indices[1:end-1]] .* ωj[:, j] ./ 2 / eom.aₑ / eom.Hₑ
        # indices for the sorted array
        Y = n2_H
        # @info X[1:argmax(X)] X[argmax(X):end]
        # @info X[1]
        try
            n2_H_dense = Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
        catch e
            n2_H_dense = Interpolate(X[1:argmax(X)], Y[1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
            f += Interpolate(X[end:-1:argmax(X)], Y[end:-1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        end
    end
    =# 
    
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
    # @show f
    return f
    
    #=
    # Third method: n=2 spotty, but not so bad, bad for n=4
    k_new = []
    f = []
    for i in 1:N
    # iterate over different oscillations
        X = a_new[indices[i]] .* ωj[i, :] ./ 2 / eom.aₑ / eom.Hₑ
        # ~ n^2 / H
        Y = [4*π/ωj[i, j]^3 * V_new[indices[i]]^2 * abs2(c_n[i, j]) / H_new[indices[i]] for j in 1:num_j]
        k_new = [k_new; X]
        f = [f; Y]
    end
    sort_ind = sortperm(k_new)
    # @show k_new[sort_ind], f[sort_ind]
    return Interpolate(k_new[sort_ind], f[sort_ind], extrapolate=LinearInterpolations.Constant(0.0)).(k)
    =#
end

function save_all(num_k, data_dir, log_k_i=0, log_k_f=2)
    @info data_dir
    eom = deserialize(data_dir * "eom.dat")

    k = @. logspace(log_k_i, log_k_f, num_k)
    f = @time get_f(eom, k)

    # mkpath(data_dir)
    npzwrite(data_dir * "spec_boltz.npz", Dict(
        "k" => k,
        "f" => f,
    ))
    return nothing
end
end
