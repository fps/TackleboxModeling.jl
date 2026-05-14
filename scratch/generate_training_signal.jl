include("generate_wave_packet.jl")

T_clicks = 5
N_clicks = 1000 * T_clicks
clicks = zeros(Float32, 48000 * T_clicks)
click_positions = Random.rand(1:(48000 * T_clicks), N_clicks)
clicks[click_positions] .= randn(Float32, N_clicks)

T_wave_packets = 10
N_wave_packets = 50 * T_wave_packets
wave_packets = zeros(Float32, 48000 * (T_wave_packets + 1))
wave_packet_positions = Random.rand(1:(48000 * T_wave_packets), N_wave_packets)
wave_packet_frequencies = max.(1, 24000 * Random.rand(N_wave_packets))
wave_packet_amplitudes = randn(N_wave_packets) ./ wave_packet_frequencies
wave_packet_periods = Random.rand(1:100, N_wave_packets)

for n in 1:N_wave_packets
  wave_packet = generate_wave_packet(wave_packet_frequencies[n], wave_packet_periods[n], 48000)
  start = wave_packet_positions[n]
  stop = start + length(wave_packet) - 1
  if stop > length(wave_packets)
    continue
  end
  wave_packets[start:stop] .+= wave_packet_amplitudes[n] .* wave_packet
end

# T = 10;
# n_samples = 48000 * T;
# signal = randn(n_samples)

# signal, fs = WAV.wavread("data/Gray_noise.wav")
# n_samples = length(signal)
#=
f_signal = FFTW.fft(signal); 
f_signal[1] = 0; f_signal; 
f_signal[2:end] ./= (1:(n_samples-1)).^(1/2)
signal = real(FFTW.ifft(f_signal)); 
=#
# signal .*= exp.(-20 .* ((1/n_samples):(1/n_samples):1))
# signal .*= ((1/n_samples):(1/n_samples):1).^4

# signal = cat(signal, clicks, dims=1)
# signal = cat(clicks, wave_packets, dims=1)
signal = wave_packets


test_file_name = "Take1_Audio 1-1_short"
test, test_fs = WAV.wavread("data/$(test_file_name).wav")

signal .*= Statistics.std(test) / Statistics.std(signal)

# signal = signal ./ maximum(abs.(signal)); 
# signal ./= Statistics.std(signal)
# f = DSP.remez(101, [(0, 3000) => 1, (10000, 24000) => 0]; Hz=48000)
# signal = DSP.filt(f, signal)
WAV.wavwrite(signal, "data/noise_input.wav"; Fs=48000)
plt(signal)
