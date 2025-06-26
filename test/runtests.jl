using GPP_Bogo_v_Boltz
using Test

#=
function _test_single_pair(f, g; atol=1e-5)
    """
    test the fourier transform with gaussian
    """
    # f(x) = cos(x)
    
    # analytical/correct results
    N = 1000
    x = range(-10, 10, N)
    y = @. f(x)
    xf, yf = GPP_Bogo_v_Boltz.Boltzmann._get_four_trafo_physics(x, y)
    
    yf_ana = g.(xf)
    
    abs_diff = abs.(yf .- yf_ana)/N
    sum_diff = sum(abs_diff[.!isnan.(abs_diff)])
    if isapprox(sum_diff, 0; atol=atol)
        return true 
    else
        # @show yf, yf_ana
        @show sum_diff
        # @show abs_diff
        return false
    end
end

@testset "boltz.jl" begin
    # step function 
    f(x) = x <= 1/2 && x >= -1/2 ? 1 : 0
    g(x) = sin(x/2) / (x/2)
    # seems step function's transform cannot be very accurate
    @test _test_single_pair(f, g, atol=1e-2)

    # Gaussian
    σ = 1.5 
    f2(x) = exp(-x^2/(2*σ^2))/sqrt(2*π*σ^2)
    g2(k) = exp(-k^2*σ^2/2)
    @test _test_single_pair(f2, g2)
end
=#

"""
helper function to test fourier series
period == 2π always
"""
function _test_f_four(f::Function, desired_coeff::Vector)
    X = range(0, 2*π, 5000)
    Y = @. f(X)
    results = GPP_Bogo_v_Boltz.Boltzmann.get_four_coeff(X, Y, 5)
    if isapprox(results, desired_coeff, atol=1e-3)
        return true 
    else 
        @show results 
        @show desired_coeff 
        return false
    end 
end

@testset "boltz.jl" begin 
    # trivial example
    @test _test_f_four(sin, [0+0im, -0.5im, 0+0im, 0+0im, 0+0im])
    @test _test_f_four(cos, [0+0im, 0.5, 0+0im, 0+0im, 0+0im])
    
    # example from wikipedia
    A = 1.0
    D = 1/3
    f(x) = x < 2*π*D ? A : 0
    c0 = A*D
    get_c(n::Int) = A/(n*π) * ( sin(2*π*n*D) - 2.0im * (sin(π*n*D))^2 )/2
    @test _test_f_four(f, [c0, get_c(1), get_c(2), get_c(3), get_c(4)])
end
