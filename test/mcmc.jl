
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::WidebandDoA.WidebandIsoIsoModel,
    mcmc ::Union{
        <: WidebandDoA.AbstractSliceSampling,
        <: WidebandDoA.AbstractMetropolis
    },
    params, data
)
    cond          = WidebandConditioned(model, data)
    params′, _, _  = ReversibleJump.transition_mcmc(rng, mcmc, cond, params)
    params′
end

@testset "WidebandIsoIso mcmc" begin
    n_snapshots  = 16
    n_sensors    = 8
    Δx           = range(0, n_sensors*0.5; length=n_sensors)
    c            = 1500
    fs           = 1000
    α, β         = 5.0, 2.0
    α_λ, β_λ     = 5.0, 2.0
    order_prior  = Dirac(1)

    model = WidebandIsoIsoModel(
        n_snapshots,
        Δx, c, fs,
        InverseGamma(α, β),
        α, β;
        order_prior,
        n_fft = n_snapshots
    )

    @testset for mcmc in [
        Slice([1.0, 1.0]),
        SliceSteppingOut([1.0, 1.0]),
        SliceDoublingOut([1.0, 1.0]),
        WidebandIsoIsoMetropolis(
            MetropolisMixture(
                IndependentMetropolis(Uniform(-π/2, π/2)),
                RandomWalkMetropolis(0.5)
            ),
            MetropolisMixture(
                IndependentMetropolis(Normal(0., 2.)),
                RandomWalkMetropolis(0.5)
            ),
        )
    ]
        @testset "determinism" begin
            n_mcmc_steps  = 10
            θ_init, y     = MCMCTesting.sample_joint(Random.default_rng(), model)
            cond          = WidebandConditioned(model, y)

            rng = StableRNG(1)
            θ   = copy(θ_init)
            for _ in 1:n_mcmc_steps
                θ, _ = ReversibleJump.transition_mcmc(rng, mcmc, cond, θ)
            end

            rng = StableRNG(1)
            θ′   = copy(θ_init)
            for _ in 1:n_mcmc_steps
                θ′, _ = ReversibleJump.transition_mcmc(rng, mcmc, cond, θ′)
            end

            @test θ == θ′
        end
        
        @testset "inference" begin
            n_pvalue_samples = 32
            n_rank_samples   = 100
            n_mcmc_steps     = 5
            n_mcmc_thin      = 1
            test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
            statistics       = θ -> begin
                stat = [only(θ).phi, only(θ).loglambda]
                vcat(stat, stat.^2)
            end
            subject = TestSubject(model, mcmc)
            
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                              statistics, show_progress=true)
        end
    end
end
