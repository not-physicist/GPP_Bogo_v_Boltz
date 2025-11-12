"""
Some convenient function to share among files/modules
"""
module Commons

# using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2
using OrdinaryDiffEq, BSplineKit

# export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct, double_trap

export logspace, get_end, check_array
export get_dϕ_SR
export get_deriv_BSpline

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
function get_deriv_BSpline(x, y, k=4)
    y_int = BSplineKit.interpolate(x, y, BSplineOrder(k))
    S = spline(y_int)
    dS = diff(S, Derivative(1))
    dydx = @. dS(x)
    return dydx
end

end
