m_over = Flux.Chain([Flux.Chain(Flux.Conv(l.weight, l.bias), x -> DSP.resample(tanh.(DSP.resample(x, 8, dims=1)), 1/8, dims=1)[:,:,:]) for l in m_min[1:(end-1)]]..., Flux.Conv(m[end].weight, m_min[end].bias)) |> cpu

WAV.wavwrite(m_over((test[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/oversampled_$(test_file_name).wav"; Fs=test_fs)

sweep, sweep_fs = WAV.wavread("data/sweep.wav")
sweep = Float32.(sweep)

WAV.wavwrite(m_over((sweep[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/oversampled_sweep.wav"; Fs=test_fs)

WAV.wavwrite(cpu(m_min((dev(sweep[:,:,:]) .- x_mean) ./ x_scale) .* y_scale .+ y_mean), "$(outpath)/sweep.wav"; Fs=test_fs)



