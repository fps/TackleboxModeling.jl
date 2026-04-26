ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "8GiB"

@info "Activating package..."

import Pkg
Pkg.activate(".")

@info "Importing packages..."
import CUDA
import cuDNN
import Flux
import DSP
import Plots
import UnicodePlots
import WAV
import Statistics
import FFTW
import Random

include("../AmpModeling.jl/src/AmpModeling.jl")

dev = Flux.gpu
cpu = Flux.cpu
# dev = cpu

@info "Loading data..."

x, fs_x = WAV.wavread("data/nam_example/input.wav")
# x = x[1:(div(size(x, 1), chunksize) * chunksize)]

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)
x .-= x_mean
x ./= x_std

UnicodePlots.lineplot(x[:], width=:auto, title="Input") |> display

y, fs_y = WAV.wavread("data/BrianMay/output_BrianMay.wav")
y = y[1:size(x,1)]

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)
y .-= y_mean
y ./= y_std

UnicodePlots.lineplot(y[:], width=:auto, title="Output") |> display

@info "Setting up model..."

AmpModeling.offset(x::Function) = 0

gains = permutedims([2^k for k in 3:5][:,:,:], (2, 1, 3)) |> dev

n_gains = length(gains)

m = Flux.Chain(
    Flux.Conv((2^7,), 1 => 1),
    x -> repeat(x, 1, n_gains, 1),
    x -> tanh.(gains .* x),
    Flux.Conv((1,), n_gains => 1),
    Flux.Conv((2^8,), 1 => 1)) |> dev

m_offset = AmpModeling.offset(m)

fft_size = 2^8

@info "Chunking data..."

chunksize = fft_size + m_offset
overlap = div(fft_size, 2)

n_samples = size(x, 1)
n_chunks = div(n_samples - chunksize, overlap)


@info "n_chunks: $n_chunks"

chunked_x = zeros(Float32, chunksize, 1, n_chunks)
chunked_y = zeros(Float32, fft_size, 1, n_chunks)

for n in 1:n_chunks
  chunked_x[:, 1, n] = x[(1+(n-1)*overlap):((n-1)*overlap+chunksize)]
  chunked_y[:, 1, n] = y[m_offset .+ ((1+(n-1)*overlap):((n-1)*overlap+fft_size))]
end

chunked_x = chunked_x[:,:,:] |> dev
chunked_y = chunked_y[:,:,:] |> dev

function stft_basis(n); exp.(-im * 2 * Float32(pi) .* (1:div(n,2)) .* (0:(n - 1))' ./ n); end

basis = stft_basis(fft_size) |> dev
window = DSP.Windows.hamming(fft_size) |> dev

function stft(x); basis * (x .* window); end

opt = Flux.setup(Flux.Adam(1e-3), m)

n_epochs = 5000

loss_min = 1f10

train_losses = []
for epoch in 1:n_epochs
    @info "Epoch: $epoch"
    global loss_min

    shift1 = Random.rand(1:fft_size)    
    shift2 = Random.rand(1:fft_size)   

    loss, grad = Flux.withgradient(m) do m
        y_hat = m(chunked_x)[:,1,:]

        fy_hat = abs.(basis * circshift(y_hat, (shift1, 0)))
        fy = abs.(basis * circshift(chunked_y[:,1,:], (shift1, 0)))

        Flux.mse(fy_hat, fy) + 1f0 * Statistics.mean(y_hat)^2
    end

    global m_prev = deepcopy(m)

    @info "loss: $loss, loss_min: $loss_min" #, grad: $(grad[1])"
    if !isfinite(loss)
        @info "Ugh"
        break
    end


    if loss < loss_min
      @info "loss_min: $loss_min"
      loss_min = loss
      global m_min = deepcopy(m)
      # WAV.wavwrite(m(dev(x[:,:,:]))[:] .* y_std |> cpu, "model_output.wav"; Fs=fs_x)
    end

    push!(train_losses, loss)
    UnicodePlots.lineplot(log10.(train_losses), width=:auto, title="Training losses (log10)") |> display
    Flux.update!(opt, m, grad[1])
end

