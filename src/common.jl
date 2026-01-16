"""
Some convenient function to share among files/modules
"""
module Commons

# using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2
using OrdinaryDiffEq, BSplineKit, Peaks, Serialization, QuadGK, ProgressBars, NPZ

# export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct, double_trap

export logspace, get_end, check_array
export get_dϕ_SR
export get_deriv_BSpline
export get_four_coeff

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
    return 10.0 .^ (range(start, stop, num))
end

"""
check if array contains nan or infinite
"""
function check_array(x::Vector)
    return any(x -> isnan(x) || !isfinite(x) ,x)
end

"""
Simple dump for struct, but instead of output to stdout, return a string for Logging
"""
function dump_struct(s)
    out = "Fields of $(typeof(s)): \n"
    for i in fieldnames(typeof(s))
        out *= "$i" * " = " * string(getfield(s, i)) * "\n"
    end
    return out
end

"""
dϕ = dϕ/dτ at slow roll trajectory in conformal time
"""
function get_dϕ_SR(dV::Real, V::Real, a::Real=1.0)
    return - a * dV / sqrt(3 * V)
end

"""
use Bspline to get derivative of dydx (uneven spacing)
depending on the input data, the order is to be adjusted

see: https://discourse.julialang.org/t/best-way-to-take-derivatives-of-unevenly-spaced-data-with-interpolations-discrete-derivatives/54097/6
"""
function get_deriv_BSpline(x, y, k=4, n=1)
    y_int = BSplineKit.interpolate(x, y, BSplineOrder(k))
    S = spline(y_int)
    dS = diff(S, Derivative(n))
    dydx = @. dS(x)
    return dydx
end

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
function get_four_coeff(num_j, t, V_ρ, dn)
    fn = dn * "four_coef.dat"
    if isfile(fn)
        return deserialize(fn)
    end

    # use minima, start from first minima
    indices = argminima(V_ρ)
    N = size(indices)[1] - 1
    
    # only keep first $whatever oscillations
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
    @time Threads.@threads for i in ProgressBar(1:N)
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
    # dm/dt / m^2
    mdm2 = get_deriv_BSpline(t[indices[1:end-1]], m_tilde, 3) ./ m_tilde .^2 
    
    #=
    # check the spline order
    x_int = range(t[indices[1]], t[indices[end-1]], length=(size(indices)[1] - 1)*10)
    y_int = S.(x_int)
    npzwrite("data/check_mdm2_spline_order.npz", Dict(
        "x" => t[indices[1:end-1]],
        "y" => m_tilde,
        "x_int" => x_int,
        "y_int" => y_int,
    ))
    # Looks good!
    =#
    
    serialize(fn, (N, indices, m_tilde, c_n, mdm2))
    npzwrite(dn * "four_coef.npz", Dict("c_n" => c_n))

    return N, indices, m_tilde, c_n, mdm2
end

end
