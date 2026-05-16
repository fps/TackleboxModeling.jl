m_over = Flux.Chain([Flux.Chain(Flux.Conv(l.weight, l.bias), x -> DSP.resample(tanh.(DSP.resample(x, 8, dims=1)), 1/8, dims=1)[:,:,:]) for l in m[1:(end-1)]]..., Flux.Conv(m[end].weight, m[end].bias)) |> cpu

WAV.wavwrite(m_over((test[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/oversampled_$(test_file_name).wav"; Fs=test_fs)

