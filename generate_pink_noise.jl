T = 10;
n_samples = 48000 * T;
signal = randn(n_samples)

# signal, fs = WAV.wavread("data/Gray_noise.wav")
# n_samples = length(signal)
#=
f_signal = FFTW.fft(signal); 
f_signal[1] = 0; f_signal; 
f_signal[2:end] ./= (1:(n_samples-1)).^(1/2)
signal = real(FFTW.ifft(f_signal)); 
=#
# signal .*= exp.(-20 .* ((1/n_samples):(1/n_samples):1))
signal .*= ((1/n_samples):(1/n_samples):1).^4
signal = signal ./ maximum(abs.(signal)); 
# f = DSP.remez(101, [(0, 3000) => 1, (10000, 24000) => 0]; Hz=48000)
# signal = DSP.filt(f, signal)
WAV.wavwrite(signal, "data/noise_input.wav"; Fs=48000)
plt(signal)
