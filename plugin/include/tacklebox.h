#pragma once

#include <FFTConvolver.h>
#include <cmath>
#include <iostream>
#include <array>

#include <hiir/PolyphaseIir2Designer.h>
#include <hiir/Upsampler2xFpu.h>
#include <hiir/Downsampler2xFpu.h>

namespace tacklebox
{
  const int n_iir_coeffs = 8;

  struct layer
  {
    std::vector<float> weights;
    float bias;
    std::string activation;
  };
  
  struct model
  {
    std::vector<layer> layers;
    float x_scale;
    float x_mean;
    float y_scale;
    float y_mean;
  };

 struct processor
  {
    std::array<std::vector<float>, 2> buffers;
    int current_buffer;
  
    float x_scale;
    float x_mean;
    float y_scale;
    float y_mean;

    std::vector<fftconvolver::FFTConvolver> convolvers;
    std::vector<float> biases;  
    std::vector<std::string> activations;
    std::vector<float> anti_derivative_buffers;
    std::array<double, n_iir_coeffs> iir_coeffs;

    std::vector<hiir::Upsampler2xFpu<n_iir_coeffs>> upsamplers;
    std::vector<hiir::Downsampler2xFpu<n_iir_coeffs>> downsamplers;

    std::vector<std::vector<float>> upsampled_input_buffers;
    std::vector<std::vector<float>> upsampled_output_buffers;

    inline int next_buffer()
    {
      return current_buffer % 2;
    }
  
    inline void tanh_activation(int const layer, int const nframes, float const bias)
    {
      std::vector<float> & in_buffer = buffers[current_buffer];

      std::vector<float> & upsampled_input_buffer = upsampled_input_buffers[layer];

      upsamplers[layer].process_block(upsampled_input_buffer.data(), in_buffer.data(), nframes);

      for (int index = 0; index < (2 * nframes); ++index)
      {
        upsampled_input_buffer[index] = tanhf(upsampled_input_buffer[index] + bias);
      }

      downsamplers[layer].process_block(in_buffer.data(), upsampled_input_buffer.data(), 2 * nframes);
    }

    inline void dist_aa_activation(int const layer, int const nframes, float const bias)
    {
      std::vector<float> & in_buffer = buffers[current_buffer];

      std::vector<float> & upsampled_input_buffer = upsampled_input_buffers[layer];
      std::vector<float> & upsampled_output_buffer = upsampled_output_buffers[layer];

      upsamplers[layer].process_block(upsampled_input_buffer.data(), in_buffer.data(), nframes);

      for (int index = 0; index < (2 * nframes); ++index)
      {
        if (index == (2 * nframes) - 1)
        {
          anti_derivative_buffers[layer] = upsampled_input_buffer[index];
        }

        const float x0 = upsampled_input_buffer[index] + bias;
        float x1 = 0;
        if (index == 0)
        {
          x1 = anti_derivative_buffers[layer] + bias;
        }
        else
        {
          x1 = upsampled_input_buffer[index - 1];
        }

        float const x0_2 = x0 * x0;
        float const x1_2 = x1 * x1;

        upsampled_output_buffer[index] = (x0 + x1) / (sqrtf(1 + x0_2) + sqrtf(1 + x1_2));
      }

      downsamplers[layer].process_block(in_buffer.data(), upsampled_output_buffer.data(), nframes);
    }

    inline void process_layer(int const layer, int const nframes)
    {
      convolvers[layer].process(buffers[current_buffer].data(), buffers[next_buffer()].data(), nframes);
      current_buffer = next_buffer();

      if (activations[layer] == "tanh")
      {
        tanh_activation(layer, nframes, biases[layer]); 
      }
      if (activations[layer] == "dist_aa")
      {
        dist_aa_activation(layer, nframes, biases[layer]); 
      }
      else
      {
        std::vector<float> out_buffer = buffers[current_buffer];
        for (int index = 0; index < nframes; ++index)
        {
          out_buffer[index] += biases[layer];
        }
      }
    }
  
    inline void process(float const * const in, float * const out, float const pre_coef, float const post_coef, int const nframes)
    {
      current_buffer = 0;

      std::vector<float> & in_buffer = buffers[current_buffer];
      for (int index = 0; index < nframes; ++index)
      {
        in_buffer[index] = (pre_coef * in[index] - x_mean) / x_scale;
      }

      for (size_t layer = 0; layer < biases.size(); ++layer)
      {
        process_layer(layer, nframes);
      }

      std::vector<float> &out_buffer = buffers[current_buffer];
      for (int index = 0; index < nframes; ++index)
      {
        out[index] = post_coef * ((out_buffer[index] * y_scale) + y_mean);
      }
    }
 
    processor(model const & m, int blocksize) :
      buffers{std::vector<float>(blocksize), std::vector<float>(blocksize)},
      current_buffer(0),
      x_scale(m.x_scale),
      x_mean(m.x_mean),
      y_scale(m.y_scale),
      y_mean(m.y_mean),
      convolvers(m.layers.size()),
      biases(m.layers.size()),
      activations(m.layers.size()),
      anti_derivative_buffers(m.layers.size(), 0),
      upsamplers(m.layers.size()),
      downsamplers(m.layers.size()),
      upsampled_input_buffers(m.layers.size()),
      upsampled_output_buffers(m.layers.size())
    {
      std::cout << "processor()...\n";

      hiir::PolyphaseIir2Designer::compute_coefs_spec_order_tbw (iir_coeffs.data(), n_iir_coeffs, 0.4);
  
      for (size_t index = 0; index < m.layers.size(); ++index)
      {
        convolvers[index].init(blocksize, m.layers[index].weights.data(), m.layers[index].weights.size());
        biases[index] = m.layers[index].bias;
        activations[index] = m.layers[index].activation;
        upsamplers[index].set_coefs(iir_coeffs.data());
        downsamplers[index].set_coefs(iir_coeffs.data());
        upsampled_input_buffers[index] = std::vector<float>(2*blocksize);
        upsampled_output_buffers[index] = std::vector<float>(2*blocksize);
      } 


      std::cout << "done.\n";
    }
  };
} 
