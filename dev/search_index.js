var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = WidebandDoA","category":"page"},{"location":"#WidebandDoA","page":"Home","title":"WidebandDoA","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for WidebandDoA.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [WidebandDoA]","category":"page"},{"location":"#WidebandDoA.WindowedSinc","page":"Home","title":"WidebandDoA.WindowedSinc","text":"WindowedSinc(n_fft)\n\nS. -C. Pei and Y. -C. Lai, \"Closed Form Variable Fractional Time Delay Using FFT,\"  in IEEE Signal Processing Letters, 2012.\n\nS. -C. Pei and Y. -C. Lai,  \"Closed form variable fractional delay using FFT with transition band trade-off,\"  IEEE International Symposium on Circuits and Systems (ISCAS), 2014.\n\nNote: Technically speaking this filter is now overkill.\n\n\n\n\n\n","category":"type"},{"location":"#WidebandDoA.array_delay","page":"Home","title":"WidebandDoA.array_delay","text":"array_delay(filter, Δn)\n\nReturns the fourier domain fractional delay filters as a matrix     H ∈ R^{ N × M × K } The fractional delay filters are the ones in:\n\n\n\n\n\n","category":"function"},{"location":"#WidebandDoA.sample_signal-Tuple{Random.AbstractRNG, WidebandDoA.WidebandNormalGammaPrior, NamedTuple}","page":"Home","title":"WidebandDoA.sample_signal","text":"sample_signal(rng, prior, params)\n\nThe sampling process is as follows:\n\nbeginaligned\n    a         sim mathcalN(0 sigma^2 Lambda) \n    epsilon sim mathcalN(0 sigma^2 mathrmI) \n    x         = H a \n    y         = x + epsilon  \nendaligned\n\nAfter marginalizing out the source signal magnitudes a, \n\nbeginaligned\n    epsilon sim mathcalN(0 sigma^2 mathrmI) \n    x         sim mathcalN(0 sigma^2 H Lambda H^top) \n    y         = x + epsilon  \nendaligned\n\nand the noise epsilon,\n\nbeginaligned\n    y         sim mathcalN(0 sigma^2 left( H Lambda H^top + mathrmI right))\nendaligned\n\nSampling from this distribution is as simple as\n\nbeginaligned\n  y = sigma H Lambda^12 z_a + sigma z_epsilon\nendaligned\n\nwhere z_a and z_epsilon are independent standard Gaussian vectors.\n\n\n\n\n\n","category":"method"}]
}
