using GPP_Bogo_v_Boltz
using Test

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
