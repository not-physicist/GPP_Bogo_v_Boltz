"""
Some convenient function to share among files/modules
"""
module Commons

# using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2
using OrdinaryDiffEq

# export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct, double_trap

export logspace, get_ρ_ϕ, get_H2_conf, get_end, check_array

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
    return 10.0 .^ (range(start, stop, num))
end

function check_array(x::Vector)
    return any(x -> isnan(x) || !isfinite(x) ,x)
end

#=
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
=#

end
