"""
module to compute exact Boltzmann result
"""
module Boltzmann


# using CubicSplines: extrapolate
using ..Commons

using LinearInterpolations, NumericalIntegration, Peaks, Statistics, Serialization, NPZ, QuadGK, ProgressBars
using BSplineKit 
using CurveFit

#=
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
=#

"""
get (complex) a single fourier coefficient using the integral formula
assume the input is exactly one period

only returns absolute value of Fourier coefficient
"""
function _get_c_n(x_min, x_max, y_itp, n::Int)
    # the period of V is always half of the oscillation frequency m
    P = x_max - x_min
    ω = 2*π*n/P
    
    res = quadgk(k -> y_itp(k)*exp(-1.0im*ω*k)/P, x_min, x_max)
    # @show res
    return abs(res[1])
end

"""
    Get Fourier series for each oscillation
    separate arrays for frequencies and four. coeffcients 
    c_n: two-d array, one for which oscillation, one for which fourier mode
"""
function get_four_coeff(num_j, t, V_ρ)
    # use minima, start from first minima
    indices = argminima(V_ρ)
    N = size(indices)[1] - 1
    
    # only keep first 50 oscillations
    # if N > 200
    #     indices = indices[1:200]
    #     N = size(indices)[1] - 1
    # end

    @info "How many minima: " size(indices)
    
    if any(x -> x < num_j, diff(indices))
        @warn "Number of points in one period could be lower than num_j; Fourier series should not be trusted in this case!" diff(indices) num_j
    end

    m_tilde = zeros(Float64, (N,))
    c_n = zeros(Float64, (N, num_j))
    Threads.@threads for i in ProgressBar(1:N)
        i1 = indices[i]
        i2 = indices[i+1]
    
        # get t and (normalized) V for a single oscillation
        t_osci = t[i1:i2] .- t[i1]
        V_osci = (V_ρ)[i1:i2]
        # P = t_osci[end] - t_osci[1]
        y_itp = BSplineKit.interpolate(t_osci, V_osci, BSplineOrder(4))
        c_n[i, :] = [_get_c_n(t_osci[1], t_osci[end], y_itp, j) for j in 1:num_j]
        m_tilde[i] = π / (t_osci[end] - t_osci[1])
        
        #=
        # check the spline order 
        if i == 1
            x_int = range(t_osci[1], t_osci[end], length=size(t_osci)[1]*10)
            y_int = y_itp.(x_int)
            npzwrite("data/check_osci_spline_order.npz", Dict(
                "x" => t_osci,
                "y" => V_osci,
                "x_int" => x_int,
                "y_int" => y_int,
            ))
        end
        # Looks good!
        =#
    end
    mpm2 = get_deriv_BSpline(t[indices[1:end-1]], m_tilde) ./ m_tilde .^2 
    
    #=
    # check the spline order
    x_int = range(t[indices[1]], t[indices[end-1]], length=(size(indices)[1] - 1)*10)
    y_int = S.(x_int)
    npzwrite("data/check_mpm2_spline_order.npz", Dict(
        "x" => t[indices[1:end-1]],
        "y" => m_tilde,
        "x_int" => x_int,
        "y_int" => y_int,
    ))
    # Looks good!
    =#

    return N, indices, m_tilde, c_n, mpm2
end


"""
compute the Boltzmann phase space distribution
decompose the oscillations of inflaton into fourier series

k in unit of aₑHₑ
"""
function get_f(eom, k::Vector, model::Symbol, dn_single=nothing)
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
    
    N, indices, m_tilde, c_n, mpm2 = @time get_four_coeff(num_j, t_new, V_new ./ ρ_new)
    # @show size(indices) size(ωj) size(ωpω2)
    
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

        # correction factor 
        C = @. 1/abs(1 + 1/j * X * eom.aₑ * H_new[1] / (a_new * H_new)[indices[1:end-1]] * (mpm2))
        # display(C)
        #=
        if j == 1
            @info "Correction factor:"
            display(X)
            display(C)
        end
        =#

        Y = n2_H .* C

        if !isnothing(dn_single) && j <= 5
            fn = dn_single * "spec_boltz_j=$j.npz"
            npzwrite(fn, Dict("X" => X, "Y" => Y))
        end

        if model == :quadratic
            n2_H_dense = Interpolate(X, Y, extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
        elseif model == :quartic
            n2_H_dense = Interpolate(X[1:argmax(X)], Y[1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
            f += n2_H_dense
            f += Interpolate(X[end:-1:argmax(X)], Y[end:-1:argmax(X)], extrapolate=LinearInterpolations.Constant(0.0)).(k)
        elseif model == :sextic
            sort_ind = sortperm(X[1:end-1])
            f += Interpolate(X[sort_ind], Y[sort_ind], extrapolate=LinearInterpolations.Constant(0.0)).(k)
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
    if single
        f, a, m = @time get_f(eom, k, model, data_dir)
    else
        f, a, m = @time get_f(eom, k, model)
    end
    
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
