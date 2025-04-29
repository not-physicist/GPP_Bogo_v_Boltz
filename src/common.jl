"""
Some convenient function to share among files/modules
"""
module Commons

# using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2
using OrdinaryDiffEq

# export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct, double_trap

export logspace, get_end, check_array
export get_dϕ_SR

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

end
