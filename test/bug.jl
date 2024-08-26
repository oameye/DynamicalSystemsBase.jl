using StochasticDiffEq, StaticArrays
using DynamicalSystemsBase, Test
using OrdinaryDiffEq: Tsit5
using StochasticDiffEq: SDEProblem, SRA, LambaEM, CorrelatedWienerProcess
using StochasticDiffEq.DiffEqNoiseProcess: CorrelatedWienerProcess
using SciMLBase: SDEProblem, AbstractSDEIntegrator, __init, SDEFunction, step!

tspan = (0., 1.)
prototype = zeros(2,2)
u0 = zeros(2)

Σ = [1.0 0.3; 0.3 1.0]
σ = 0.2

f(u, p, t) = σ .* u

g1(u, p, t) = σ .* ones(2)

W0 = SVector{length(u0)}(zeros(eltype(u0), length(u0)))
W = CorrelatedWienerProcess(Σ,0.0,W0,W0)

prob = SDEProblem(f, g1, u0, tspan; noise=W)

integ = __init(
    prob,
    LambaEM();
    # Integrators are used exclusively iteratively. There is no reason to save anything.
    save_start = false,
    save_end = false,
    save_everystep = false,
    # DynamicalSystems.jl operates on integrators and `step!` exclusively,
    # so there is no reason to limit the maximum time evolution
    maxiters = Inf
)
step!(integ, 1.0)

sde = CoupledSDEs(f, u0; covariance = Σ)
step!(sde, 1.0)

###########################################################################################
###########################################################################################

@inbounds function lorenz_rule(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = [0, 10.0, 0]
g(u, p, t) = SVector{length(u0)}(ones(eltype(u0), length(u0)))
p0 = [10, 28, 8 / 3]
Γ = [1.0 0.3 0.0; 0.3 1 0.5; 0.0 0.5 1.0]

W0 = SVector{length(u0)}(zeros(eltype(u0), length(u0)))
W = CorrelatedWienerProcess(Γ,0.0,W0,W0)

prob = SDEProblem(lorenz_rule, g, u0, tspan, p0; noise=W)

integ = __init(
    prob,
    LambaEM();
    # Integrators are used exclusively iteratively. There is no reason to save anything.
    save_start = false,
    save_end = false,
    save_everystep = false,
    # DynamicalSystems.jl operates on integrators and `step!` exclusively,
    # so there is no reason to limit the maximum time evolution
    maxiters = Inf
)
step!(integ, 1.0)


lor_oop_cov = CoupledSDEs(lorenz_rule, u0, p0; covariance = Γ)
step!(lor_oop_cov, 1.0)

###########################################################################################
###########################################################################################

function lorenz_rule(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return [du1, du2, du3]
end

u0 = [0, 10.0, 0]
g(u, p, t) = ones(3)
p0 = [10, 28, 8 / 3]
Γ = [1.0 0.3 0.0; 0.3 1 0.5; 0.0 0.5 1.0]

W = CorrelatedWienerProcess(Γ,0.0,zeros(3),zeros(3))

prob = SDEProblem(lorenz_rule, g, u0, tspan, p0; noise=W)
solve(prob, LambaEM())

integ = __init(
    prob,
    LambaEM();
    # Integrators are used exclusively iteratively. There is no reason to save anything.
    save_start = false,
    save_end = false,
    save_everystep = false,
    # DynamicalSystems.jl operates on integrators and `step!` exclusively,
    # so there is no reason to limit the maximum time evolution
    maxiters = Inf
)
step!(integ, 1.0)
