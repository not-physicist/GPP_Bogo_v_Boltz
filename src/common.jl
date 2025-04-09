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

"""
energy density of inflaton field
in conformal time (dϕ = dϕdτ)
"""
function get_ρ_ϕ(ϕ, dϕ, a, V)
    return dϕ^2 / (2*a^2) + V(ϕ)
end

"""
conformal Hubble squared  
"""
function get_H2_conf(ϕ, dϕ, a, ρ_r, V)
    ρ_ϕ = get_ρ_ϕ(ϕ, dϕ, a, V)
    return a^2 * (ρ_r + ρ_ϕ) / 3.
end


#=
"""
struct to store the ODE data;
note that they may have different length (due to the derivatives)
"""
struct ODEData{V<:Vector, F<:Real}
    τ::V
    ϕ::V
    dϕ::V
    a::V
    app_a::V
    H::V
    err::V

    aₑ::F
    Hₑ::F
end

"""
read ODE solution stored in data/ode.npz
"""
function read_ode(data_dir::String)::ODEData
    # maybe a try catch clause here; not sure if necessary
    fn = data_dir * "ode.npz"
    data = npzread(fn)
    #  fn = data_dir * "ode.jld2"
    #  data = load(fn)

    τ = data["tau"]
    ϕ = data["phi"]
    dϕ = data["phi_d"]
    a = data["a"]
    app_a = data["app_a"]
    H = data["H"]
    err = data["err"]
    aₑ = data["a_end"]
    Hₑ = data["H_end"]
    return ODEData(τ, ϕ, dϕ, a, app_a, H, err, aₑ, Hₑ)
end

"""
get scale factor and conformal time at the end of inflation
can actually replace scale factor witn any other quantity
"""
function get_end(ϕ::Vector, dϕ::Vector, a::Vector, τ::Vector, ϕₑ::Real)
    # generate an appropriate mask
    flag = true  # whether dϕ has changed sign 
    i = 1  # index for while
    mask = zeros(Int, size(ϕ))
    while flag == true && i < size(ϕ)[1]
        mask[i] = 1
        i += 1
        if dϕ[i] * dϕ[2] < 0
            # terminate after sign change
            # for whatever reason dϕ[1] is always 0 
            flag = false
        end
    end
    mask = BitVector(mask)
    
    # depending small/large field model, the field value array can be descending or ascending
    try
        itp = LinearInterpolations.Interpolate((ϕ[mask],), τ[mask])
    catch
        itp = LinearInterpolations.Interpolate((reverse(ϕ[mask]),), reverse(τ[mask]))
    end
    τₑ = itp(ϕₑ)
    
    itp = LinearInterpolations.Interpolate((τ[mask],), a[mask])
    aₑ = itp(τₑ)
    return τₑ, aₑ
end
=#

function get_end(sol::SciMLBase.ODESolution)
    _a(t) = sol(t, Val{0}, idxs=3)
    _ap(t) = sol(t, Val{1}, idxs=3)
    _app(t) = sol(t, Val{2}, idxs=3)

    _ϵ₁(t) = @. 2 - _app(t) * _a(t) / _ap(t)^2
    # normal Hubble
    _H(t) = @. _ap(t) / _a(t)^2
   
    end_i = findfirst(x -> x >= 1.0, _ϵ₁(sol.t))
    # @show _ϵ₁(sol.t[end_i]), log(_a(sol.t[end_i])), _H(sol.t[end_i])
    a_end = _a(sol.t[end_i])
    H_end = _H(sol.t[end_i])

    # return _ϵ₁(sol.t)
    return a_end, H_end
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
