sweep, sweep_fs = WAV.wavread("data/sweep.wav")
sweep = Float32.(sweep)

WAV.wavwrite(cpu(m_min((dev(sweep[:,:,:]) .- x_mean) ./ x_scale) .* y_scale .+ y_mean), "$(outpath)/sweep.wav"; Fs=test_fs)

f = DSP.SecondOrderSections(DSP.digitalfilter(DSP.Lowpass(0.5), DSP.Chebyshev2(8, 60)))

function oversampled_dist_aa2(x)
  x = 2 * repeat(x, inner=(2,1,1))
  x[1:2:end, :, :] .= 0
  x = DSP.filt(f, x)
  x = dist_aa2(x)
  x = DSP.filt(f, x)
  x[2:2:end, :,:]
end

m_over_iir = Flux.Chain([Flux.Chain(Flux.Conv(l[1].weight, l[1].bias), x -> oversampled_dist_aa2(x)) for l in m_min[1:(end-1)]]..., Flux.Conv(m[end][1].weight, m_min[end][1].bias)) |> cpu

@info("iir oversampling")

WAV.wavwrite(m_over_iir((sweep[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/2x_iir_oversampled_sweep.wav"; Fs=test_fs)


for oversampling in [2, 4, 8]
  @info "oversampling $(oversampling)x"
  m_over = Flux.Chain([Flux.Chain(Flux.Conv(l[1].weight, l[1].bias), x -> DSP.resample(activation(DSP.resample(x, oversampling, dims=1)), 1/oversampling, dims=1)[:,:,:]) for l in m_min[1:(end-1)]]..., Flux.Conv(m[end][1].weight, m_min[end][1].bias)) |> cpu

  # WAV.wavwrite(m_over((test[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/oversampled_$(test_file_name).wav"; Fs=test_fs)

  WAV.wavwrite(m_over((sweep[:,:,:] .- x_mean) ./ x_scale) .* y_scale .+ y_mean, "$(outpath)/$(oversampling)x_oversampled_sweep.wav"; Fs=test_fs)
end


