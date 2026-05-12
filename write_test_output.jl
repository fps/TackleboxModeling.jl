@info "Writing test file in $(outpath),,,"
m_offset = 48000 - size(m(dev(zeros(Float32, 48000, 1, 1))), 1)

# test_out = m_min(dev(test./ Statistics.std(test))[:,:,:])[:] .* y_scale |> cpu
test_out = m_min(dev((test .- x_mean) ./ x_scale)[:,:,:])[:] .* y_scale .+ y_mean |> cpu
test_out = cat(zeros(Float32, m_offset), test_out, dims=1)
WAV.wavwrite(test_out, "$(outpath)/test_$(test_file_name).wav"; Fs=fs_x)

train_out = m_min(dev(x)[:,:,:])[:] .* y_scale .+ y_mean |> cpu
train_out = cat(zeros(Float32, m_offset), train_out, dims=1)
WAV.wavwrite(train_out, "$(outpath)/train_out.wav"; Fs=fs_x)

