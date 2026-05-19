sweep, sweep_fs = WAV.wavread("data/sweep.wav")
sweep = Float32.(sweep)

WAV.wavwrite(cpu(m_min((dev(sweep[:,:,:]) .- x_mean) ./ x_scale) .* y_scale .+ y_mean), "$(outpath)/sweep.wav"; Fs=test_fs)

for oversampling in [2, 4, 8]
  m_over = Flux.Chain([Flux.Chain(Flux.Conv(l[1].weight, l[1].bias), x -> DSP.resample(activation(DSP.resample(x, oversampling, dims=1)), 1/oversampling, dims=1)[:,:,:]) for l in m_min[1:(end-1)]]..., Flux.Conv(m[end][1].weight, m_min[end][1].bias)) |> cpu

  # WAV.wavwrite(m_over((test[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/oversampled_$(test_file_name).wav"; Fs=test_fs)

  WAV.wavwrite(m_over((sweep[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/$(oversampling)x_oversampled_sweep.wav"; Fs=test_fs)
end




