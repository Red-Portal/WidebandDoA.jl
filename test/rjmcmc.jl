
function MCMCTesting.sample_joint(
    rng  ::Random.AbstractRNG,
    model::WidebandDoA.WidebandNormalGammaPrior
)
    all_params = WidebandDoA.sample_params(rng, model)
    y          = WidebandDoA.sample_signal(rng, model, all_params)
    params     = @. WidebandDoA.WidebandNormalGammaParam(all_params.phi, log(all_params.lambda))
    params, y
end

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
    n_snapshots  = 16
    n_sensors    = 8
    Δx           = range(0, n_sensors*0.5; length=n_sensors)
    c            = 1500
    fs           = 1000
    delay_filter = WidebandDoA.WindowedSinc(n_snapshots)
    α, β         = 5.0, 5.0
    α_λ, β_λ     = 5.0, 5.0
    order_prior  = truncated(Poisson(2), 0, 4)

    prior = WidebandDoA.WidebandNormalGammaPrior(
        n_snapshots, delay_filter, Δx, c, fs, order_prior, α, β, α_λ, β_λ
    )

    prop  = UniformNormalLocalProposal(0.0, 1.0)
    mcmc  = SliceSteppingOut([1.0, 1.0])


    @testset for jump in [
        #AnnealedJumpProposal(4, prop, ArithmeticPath()),
        AnnealedJumpProposal(4, prop, GeometricPath()),
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
            n_mcmc_steps     = 10
            n_mcmc_thin      = 4
            test             = ExactRankTest(n_rank_samples, n_mcmc_steps, n_mcmc_thin)
            statistics       = θ -> [length(θ)]
            
            subject = TestSubject(prior, rjmcmc)
            
            @test seqmcmctest(test, subject, 0.001, n_pvalue_samples;
                              statistics, show_progress=true)
        end
    end
end
