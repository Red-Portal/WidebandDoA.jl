
function subbanddas(
    R      ::Array,
    θ_range::AbstractVector,
    f_range::AbstractVector,
    conf   ::ArrayConfig
)
    mapreduce(hcat, enumerate(f_range)) do (j, f)
        Af = steering_matrix(θ_range, f_range[j], conf)
        Rf = view(R, :, :, j)
        real.(sum(conj(Af).*(Rf*Af), dims=1)[1,:])
    end
end

function subbandmvdr(
    R        ::Array,
    θ_range  ::AbstractVector,
    f_range  ::AbstractVector,
    conf     ::ArrayConfig;
    dl_factor::Real = 1e-1/size(R,1)
)
    mapreduce(hcat, enumerate(f_range)) do (j, f)
        Af = steering_matrix(θ_range, f_range[j], conf)
        Rf = view(R, :, :, j)

        # Diagonal loading.
        # Factor recommendd by Featherstone et al. (1997)
        Rf_smooth_dl = Rf + tr(Rf)*dl_factor*I

        ARinvA = sum(conj(Af).*(Rf_smooth_dl\Af), dims=1)[1,:]
        @. 1 / real(ARinvA)
    end
end

function cfar(
    power  ::AbstractVector,
    n_guard::Int,
    n_train::Int;
    false_alarm_rate::Real = 0.1,
    gain                   = nothing,
)
    N = length(power)
    peak_indices, peak_powers = Peaks.findmaxima(power)
    sum(zip(peak_indices, peak_powers)) do (peak_idx, peak_power)
        avg    = 0.0
        n_data = 1

        # Left train cells
        for j in (peak_idx - n_guard - n_train):(peak_idx - n_guard - 1)
            if j ≥ 1 && j ≤ N
                avg     = (n_data-1)/(n_data)*avg + power[j]/n_data
                n_data += 1
            end
        end

        # Right train cells
        for j in (peak_idx + n_guard + 1):(peak_idx + n_guard + n_train)
            if j ≥ 1 && j ≤ N
                avg     = (n_data-1)/(n_data)*avg + power[j]/n_data
                n_data += 1
            end
        end

        gain = if isnothing(gain)
            n_data*(false_alarm_rate^(-1/n_data) - 1)
        else
            gain
        end
        thres = gain*avg
        peak_power ≥ thres
    end
end
