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

plt(x) = UnicodePlots.lineplot(x[:] |> cpu, width=:auto)
plt(x, title) = UnicodePlots.lineplot(x[:] |> cpu, width=:auto, title=title)

@info "Loading data..."

# x, fs_x = WAV.wavread("data/Take1_Audio 1-1_short.wav")
x, fs_x = WAV.wavread("data/noise_input.wav")
# x = x[1:(div(size(x, 1), chunksize) * chunksize)]

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)
x .-= x_mean
x ./= x_std

# x = x[1:div(size(x, 1), 4)]

plt(x, "Input") |> display

# outpath = "data/BrianMay"
outpath = "data/marshall bluesbreaker 1962"
# outpath = "data/nam_example"

# y, fs_y = WAV.wavread("$(outpath)/nam_Take1_Audio 1-1_short.wav")
y, fs_y = WAV.wavread("$(outpath)/noise_output.wav")
y = y[1:size(x,1)]

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)
y .-= y_mean
y ./= y_std

plt(y, "Output") |> display

test_file_name = "Take1_Audio 1-1_short"
test, test_fs = WAV.wavread("data/$(test_file_name).wav")

plt(test, "Test") |> display

#=
@info "Filtering data..."

pb = 16000
sb = 18000
f = DSP.remez(DSP.remezord(pb/fs_x, sb/fs_x, 0.01, 0.0001), [(0, pb/fs_x) => 1, (sb/fs_x, 0.5) => 0])

x = DSP.filt(f, x)
y = DSP.filt(f, y)
=#

@info "Setting up model..."

AmpModeling.offset(x::Function) = 0

gains = permutedims([2^k for k in 3:9][:,:,:], (2, 1, 3)) |> dev

n_gains = length(gains)

# f_decimation = DSP.remez(DSP.remezord(16000/(48000*8), 20000/(48000*8), 0.1, 0.001), [(0, 20000) => 1, (22000, 48000*8/2) => 0], Hz=(48000*8))

m = Flux.Chain(
    # Flux.Conv((2^7,), 1 => 1, Flux.tanh),
    Flux.Conv((2^3,), 1 => n_gains),
    # x -> repeat(x, 1, n_gains, 1),
    # Flux.Upsample((2,), :bilinear),
    # Flux.Upsample((2,), :bilinear),
    # Flux.Upsample((2,), :bilinear),
    x -> tanh.(gains .* x),
    # Flux.MeanPool((2,), stride=2),
    # Flux.MeanPool((2,), stride=2),
    # Flux.MeanPool((2,), stride=2),
    Flux.Conv((1,), n_gains => 1),
    Flux.Conv((2^3,), 1 => 1)
    # [Flux.Conv((16,), 1 => 1, dilation = 4^k) for k in 0:3]...
) |> dev

# m[1].weight .*= 32

function stft_basis(n); exp.(-im * 2 * Float32(pi) .* (1:div(n,2)) .* (0:(n - 1))' ./ n); end

k_max = 11
fft_sizes = [2^k_max 2^(k_max-1) 2^(k_max-2)]
# fft_sizes = [2^k_max]
  
@info "fft_sizes: $fft_sizes"
bases = map(fft_size -> stft_basis(fft_size), fft_sizes) |> dev
windows = map(fft_size -> DSP.Windows.hann(fft_size), fft_sizes) |> dev
bases = map(n -> windows[n]' .* bases[n], 1:length(bases))
# window ./= sum(window)

function stft(x); basis * (x .* window); end

n_epochs = 200

loss_min = 1f10

patience = 2^6

min_epoch = 1

lr = 1f-3

function stft_loss(overlap, bases, fft_sizes, y, y_hat)
  l = 0

  n_y_hat = size(y_hat, 1)

  for n in 1:length(bases)  
    offset = Random.rand(1:fft_sizes[n])
    n_base = size(bases[n], 1)
    fy = abs.(bases[n] * y[(1:fft_sizes[n]) .+ offset, 1, :]) # ./ dev(reverse((1:n_base)))
    fy_hat = abs.(bases[n] * y_hat[(1:fft_sizes[n]) .+ offset, 1, :]) # ./ dev(reverse((1:n_base)))

    l += Flux.mse(fy, fy_hat) ./ Statistics.mean(fy.^2)
    # l += Statistics.mean(abs.(log.(fy) .- log.(fy_hat)))
  end
  l / length(bases)
end

train_losses = []
for stage in 1:7
  global m
  m_offset = 48000 - size(m(dev(zeros(Float32, 48000, 1, 1))), 1)
  
  @info "m_offset: $m_offset"
  
  
  @info "Chunking data..."
  
  chunksize_x = maximum(fft_sizes) * 2 + m_offset
  chunksize_y = maximum(fft_sizes) * 2
  overlap = div(minimum(fft_sizes), 2)
  
  n_samples = size(x, 1)
  n_chunks = div(n_samples - chunksize_x, overlap)
  
  batchsize = min(2^12, n_chunks)

  
  @info "n_chunks: $n_chunks"
  
  chunked_x = zeros(Float32, chunksize_x, 1, n_chunks)
  chunked_y = zeros(Float32, chunksize_y, 1, n_chunks)
  
  for n in 1:n_chunks
    chunked_x[:, 1, n] = x[(1+(n-1)*overlap):((n-1)*overlap+chunksize_x)]
    chunked_y[:, 1, n] = y[m_offset .+ ((1+(n-1)*overlap):((n-1)*overlap+chunksize_y))]
  end
  
  chunked_x = chunked_x[:,:,:] |> dev
  chunked_y = chunked_y[:,:,:] |> dev
  
  opt = Flux.setup(Flux.Adam(lr), m)

  for epoch in 1:n_epochs
      if epoch <= 10
        Flux.adjust!(opt, lr / (11 - epoch))
      end
      @info "Epoch: $epoch"
      global loss_min
      global min_epoch
  
      losses = []
      for (x,y) in Flux.MLUtils.DataLoader((chunked_x, chunked_y), batchsize=batchsize, shuffle=true)
          # print(".")
          loss, grad = Flux.withgradient(m) do m
              y_hat = m(x)
      
              # fy_hat = abs.(basis * (window .* circshift(y_hat, (shift1, 0))))
              # fy = abs.(basis * (window .* circshift(y[:,1,:], (shift1, 0)))) 
      
              # fy_hat = abs.(basis * (window .* y_hat))
              # fy = abs.(basis * (window .* y[:,1,:]))
      
              # fy_hat = abs.(basis * y_hat) ./ fft_size
              # fy = abs.(basis * y[:,1,:]) ./ fft_size
      
              #=
              fym = Statistics.mean(fy)
      
              fy = fy ./ fym
              fy_hat = fy_hat ./ fym
              =#
      
              # Flux.mse(fy_hat, fy) ./ Statistics.mean(fy.^2) + 1f-2 * Statistics.mean(y_hat)^2
              stft_loss(overlap, bases, fft_sizes, y, y_hat) + Statistics.mean(y_hat).^2
          end
          Flux.update!(opt, m, grad[1])
          push!(losses, loss)
      end
  
      loss = Statistics.mean(losses)
  
      global m_prev = deepcopy(m)
  
      @info "loss: $loss, loss_min: $loss_min, min_epoch: $(min_epoch), (epoch - min_epoch): $(epoch - min_epoch)" #, grad: $(grad[1])"
      if !isfinite(loss)
          @info "Ugh"
          break
      end
  
      if loss < loss_min
        @info "loss_min: $loss_min"
        loss_min = loss
        min_epoch = epoch
        global m_min = deepcopy(m)
        # WAV.wavwrite(m(dev(x[:,:,:]))[:] .* y_std |> cpu, "model_output.wav"; Fs=fs_x)
      end
  
      push!(train_losses, loss)
      plt(log10.(train_losses), "Training losses (log10)") |> display
  
      if epoch - min_epoch > patience
        @info "Patience exhausted: $(epoch - min_epoch)"
        break
      end
  end

  if stage <= 6
    m = Flux.Chain(Flux.Conv(cat(m[1].weight, 1f-10 .* randn(Float32, size(m[1].weight)...), dims=1), m[1].bias), m[2], m[3],  Flux.Conv(cat(m[4].weight, 1f-10 .* randn(Float32, size(m[4].weight)...), dims=1), m[4].bias)) |> dev
  else
    m = Flux.Chain(m[1], m[2], m[3],  Flux.Conv(cat(m[4].weight, 1f-10 .* randn(Float32, size(m[4].weight)...), dims=1), m[4].bias)) |> dev
  end
end
  
include("write_test_output.jl")

#=
@info "Writing test file in $(outpath),,,"

test_out = m_min(dev(test ./ x_std)[:,:,:])[:] .* y_std |> cpu
test_out = cat(zeros(Float32, m_offset), test_out, dims=1)
WAV.wavwrite(test_out, "$(outpath)/test_$(test_file_name).wav"; Fs=fs_x)

=#
