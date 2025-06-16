"""
module to compute exact Boltzmann result
"""
module Boltzmann

using Base: absdiff
using FFTW, CubicSplines

"""
(Physicists') Fourier transform of discrete data points (x, y)
Re-sample the input using interpolation
Assume real input
"""
function _get_four_trafo_physics(x, y)
    Δx = minimum(diff(x))
    x_new = range(x[1], x[end], step=Δx)
    intp = CubicSpline(x, y)
    y_new = intp[x_new]

    yf = rfft(y_new)
    xf = rfftfreq(length(y_new)) * (2*π) / Δx
    yf_new = @. yf * Δx * exp(-1.0im*xf*x[1])

    return xf, yf_new
end

"""
compute phase space distribution of graviton 
exact Boltzmann method.
"""
function get_f(eom)

    ω, V_tilde = _get_four_trafo_physics(eom.t, eom.V)
end

end
