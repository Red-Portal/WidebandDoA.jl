
function MCMCTesting.markovchain_transition(
    rng   ::Random.AbstractRNG,
    model ::WidebandDoA.WidebandNormalGammaPrior,
    rjmcmc::ReversibleJumpMCMC,
    θ, y
)
    joint = WidebandNormalGamma(y, model)
    _, init_state = AbstractMCMC.step(
        rng, joint, rjmcmc; initial_params=θ, initial_order=length(θ)
    )
    _, state = AbstractMCMC.step(
        rng, joint, rjmcmc, init_state
    )
    state.param
end

@testset "WidebandNormalGamma rjmcmc" begin
    n_snapshots  = 32
    n_sensors    = 20
    Δx           = range(0, n_sensors*0.5; length=n_sensors)
    c            = 1500
    fs           = 1000
    delay_filter = WidebandDoA.WindowedSinc(n_snapshots)
    α, β         = 5.0, 2.0
    α_λ, β_λ     = 5.0, 2.0
    order_prior  = Poisson(2)

    prior = WidebandDoA.WidebandNormalGammaPrior(
        n_snapshots, delay_filter, Δx, c, fs, order_prior, α, β, α_λ, β_λ
    )

    prop  = UniformNormalLocalProposal(0.0, 1.0)
    mcmc  = SliceSteppingOut([1.0, 1.0])

    @testset for jump in [
        AnnealedJumpProposal(prop, ArithmeticPath(4)),
        AnnealedJumpProposal(prop, GeometricPath(4)),
        IndepJumpProposal(prop)
    ]
        rjmcmc = ReversibleJumpMCMC(order_prior, jump, mcmc)

        @testset "determinism" begin
            n_mcmc_steps = 10
            θ_init, y    = MCMCTesting.sample_joint(Random.default_rng(), prior)
            joint        = WidebandNormalGamma(y, prior)

            rng     = StableRNG(1)
            samples = AbstractMCMC.sample(
                rng, joint, rjmcmc, n_mcmc_steps; 
                initial_params = copy(θ_init),
                initial_order  = length(θ_init)
            )

            rng     = StableRNG(1)
            samples′ = AbstractMCMC.sample(
                rng, joint, rjmcmc, n_mcmc_steps; 
                initial_params = copy(θ_init),
                initial_order  = length(θ_init)
            )

            @test samples == samples′
        end
        
        @testset "inference" begin
            n_pvalue_samples = 32
            n_rank_samples   = 100
            n_mcmc_steps     = 5
            n_mcmc_thin      = 1
            test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
            statistics       = θ -> [length(θ)]
            
            subject = TestSubject(prior, rjmcmc)
            
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                              statistics, show_progress=true)
        end
    end
end
