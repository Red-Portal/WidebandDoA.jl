
function MCMCTesting.markovchain_transition(
    rng  ::Random.AbstractRNG,
    model::WidebandDoA.WidebandNormalGammaPrior,
    mcmc ::WidebandDoA.AbstractSliceSampling,
    θ, y
)
    joint = WidebandNormalGamma(y, model)
    θ, _  = ReversibleJump.transition_mcmc(rng, mcmc, joint, θ)
    θ
end

@testset "WidebandNormalGamma mcmc" begin
    n_snapshots  = 16
    n_sensors    = 8
    Δx           = range(0, n_sensors*0.5; length=n_sensors)
    c            = 1500
    fs           = 1000
    delay_filter = WidebandDoA.WindowedSinc(n_snapshots)
    α, β         = 5.0, 2.0
    α_λ, β_λ     = 5.0, 2.0
    order_prior  = Dirac(1)

    prior = WidebandDoA.WidebandNormalGammaPrior(
        n_snapshots, delay_filter, Δx, c, fs, order_prior, α, β, α_λ, β_λ
    )

    @testset for mcmc in [
        SliceSteppingOut([1.0, 1.0])
    ]
        @testset "determinism" begin
            n_mcmc_steps = 10
            θ_init, y    = MCMCTesting.sample_joint(Random.default_rng(), prior)
            joint        = WidebandNormalGamma(y, prior)

            rng = StableRNG(1)
            θ   = copy(θ_init)
            for _ in 1:n_mcmc_steps
                θ, _ = ReversibleJump.transition_mcmc(rng, mcmc, joint, θ)
            end

            rng = StableRNG(1)
            θ′   = copy(θ_init)
            for _ in 1:n_mcmc_steps
                θ′, _ = ReversibleJump.transition_mcmc(rng, mcmc, joint, θ′)
            end

            @test θ == θ′
        end
        
        @testset "inference" begin
            n_pvalue_samples = 32
            n_rank_samples   = 100
            n_mcmc_steps     = 10
            n_mcmc_thin      = 1
            test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
            statistics       = θ -> begin
                stat = [first(θ).phi, first(θ).loglambda]
                vcat(stat, stat.^2)
            end
            
            subject = TestSubject(prior, mcmc)
            
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                              statistics, show_progress=true)
        end
    end
end
