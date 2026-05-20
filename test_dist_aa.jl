sweep, sweep_fs = WAV.wavread("data/sweep.wav")
sweep = Float32.(sweep)

WAV.wavwrite(dist.(10 .* sweep)[:], "data/sweep_dist.wav"; Fs=sweep_fs)
WAV.wavwrite(dist_aa(10 .* sweep)[:], "data/sweep_dist_aa.wav"; Fs=sweep_fs)
WAV.wavwrite(dist_aa2(10 .* sweep)[:], "data/sweep_dist_aa2.wav"; Fs=sweep_fs)

for oversampling in [2 4 8]
  WAV.wavwrite(DSP.resample(dist.(DSP.resample(10 .* sweep, oversampling, dims=1))[:], 1/oversampling, dims=1), "data/sweep_dist_$(oversampling)x.wav"; Fs=sweep_fs)
  WAV.wavwrite(DSP.resample(dist_aa(DSP.resample(10 .* sweep, oversampling, dims=1))[:], 1/oversampling, dims=1), "data/sweep_dist_aa_$(oversampling)x.wav"; Fs=sweep_fs)
  WAV.wavwrite(DSP.resample(dist_aa2(DSP.resample(10 .* sweep, oversampling, dims=1))[:], 1/oversampling, dims=1), "data/sweep_dist_aa2_$(oversampling)x.wav"; Fs=sweep_fs)
end


