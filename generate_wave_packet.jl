function generate_wave_packet(f, n_periods, fs)
  n = n_periods * Int(round(fs / f))
  DSP.Windows.gaussian(n, 1/9) .* sin.((2 * pi * f / fs) .* (0:(n-1)))
end
