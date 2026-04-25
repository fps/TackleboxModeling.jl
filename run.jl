ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

@info "Activating package..."

import Pkg
Pkg.activate(".")

@info "Importing packages..."
import CUDA
import cuDNN
# import cuFFT
import Flux
import DSP
import Plots
import UnicodePlots
import WAV
import Statistics
import FFTW

dev = Flux.gpu
cpu = Flux.cpu
# dev = cpu

@info "Loading data..."

x, fs_x = WAV.wavread("data/nam_example/input.wav")
x = Float32.(x[40000:200000])
x = x[1:(div(size(x, 1), 8) * 8)]
# x = x[1:(div(size(x, 1), 8))]
# x = randn(2^12)

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)
x .-= x_mean
x ./= x_std

UnicodePlots.lineplot(x[:], width=:auto, title="Input") |> display

y, fs_y = WAV.wavread("data/nam_example/output.wav")
y = Float32.(y[40000:200000])
# y = y[1:(div(size(x, 1), 8) * 8)]
y = y[1:size(x,1)]
# y = randn(2^12)
# y = DSP.resample(DSP.resample(y, 1/4), 4)

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)
y .-= y_mean
y ./= y_std

UnicodePlots.lineplot(y[:], width=:auto, title="Output") |> display

y = dev(y)[:,:,:]

f = DSP.digitalfilter(DSP.Highpass(350/fs_x), DSP.FIRWindow(DSP.Windows.hamming(2^9+1)))

x = DSP.filt(f, x)

gains = permutedims([2^k for k in 3:5][:,:,:], (2, 1, 3)) |> dev

x = dev(x)[:,:,:]

# x_expanded = Float32.(cat([tanh.(g .* x) for g in gains]..., dims=2)) |> dev
# x_expanded = x_expanded[:,:,:]
n_gains = length(gains)

m = Flux.Chain(
    # Flux.Conv((4,), 1 => 1),
    x -> repeat(x, 1, n_gains, 1),
    x -> tanh.(gains .* x),
    Flux.Conv((1,), n_gains => 1),
    Flux.Conv((128,), 1 => 1)) |> dev

# m = Flux.Conv((1,), 1=>1) |> dev
opt = Flux.setup(Flux.Adam(1e-2), m)

n_epochs = 1000

# function stft(x); [ sum(x .* exp.(-im * 2 * Float32(pi) * f .* (1:size(x, 1)))) for f in Float32.((0:size(x, 1))./size(x,1)) ]; end
function stft_basis(n); exp.(-im * 2 * Float32(pi) .* (1:n) .* (0:(n - 1))' ./ n); end
fft_size = 2^8

basis = stft_basis(fft_size) |> dev
window = DSP.Windows.hamming(fft_size) |> dev

function stft(x); basis * (x .* window); end

function spectrogram(x, n, advance=div(n,1))
    N = size(x, 1)
    n_chunks = div(N - n, advance)
    out = zeros(Float32, n, n_chunks) |> dev

    for k in 1:n_chunks
        out = cat([abs.(stft(x[(1+(k-1)*advance):((k-1)*advance+n)])) ./ (n_chunks * sqrt(n)) for k in 1:n_chunks ]..., dims=2)
    end

    out[1:div(size(out, 1), 2), :]
end

function spectrum(x, n, advance=div(n,2))
    N = size(x, 1)
    n_chunks = div(N - n, advance)
    out = zeros(Float32, n) |> dev

    for k in 1:n_chunks
        out = out + abs.(stft(x[(1+(k-1)*advance):((k-1)*advance+n)])) ./ (n_chunks * sqrt(n))
    end

    out[1:div(size(out, 1), 2)]
end


train_losses = []
for epoch in 1:n_epochs
    @info "Epoch: $epoch"
    # global grad, loss
    loss, grad = Flux.withgradient(m) do m
        y_hat = m(x)
        y2 = y[(size(y, 1)-size(y_hat, 1)+1):end, :, :]

        # fy_hat = FFTW.fft(y_hat)[2:div(size(y_hat, 1), 2),:,:]
        # fy2 = FFTW.fft(y2)[2:div(size(y2, 1), 2),:,:]

        fy_hat = spectrum(y_hat[:], fft_size) 
        fy2 = spectrum(y2[:], fft_size) 

        fy_hat = spectrogram(y_hat[:], fft_size) 
        fy2 = spectrogram(y2[:], fft_size) 

        Flux.mse(fy_hat, fy2)
        # Flux.mse(y_hat, y2)
    end
    @info "loss: $loss" #, grad: $(grad[1])"
    if !isfinite(loss)
        @info "Ugh"
        break
    end
    push!(train_losses, loss)
    UnicodePlots.lineplot(log10.(train_losses), width=:auto, title="Training losses (log10)") |> display
    # UnicodePlots.lineplot(log10.(spectrum(m(x), fft_size)[:] |> cpu), width=:auto, title="m(x)") |> display
    # UnicodePlots.lineplot(log10.(spectrum(y, fft_size)[:] |> cpu), width=:auto, title="y") |> display
    WAV.wavwrite(m(x)[:] .* y_std |> cpu, "model_output.wav"; Fs=fs_x)

    Flux.update!(opt, m, grad[1])
end
