#pragma once

#include <FFTConvolver.h>
#include <cmath>
#include <iostream>
#include <array>

#include <oversample.h>

namespace tacklebox
{
  const int n_iir_coeffs = 32;

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
    std::vector<float> dist_aa_buffers;
    std::vector<std::array<float, 2>> dist_aa2_buffers;

    std::vector<oversample2x> oversamplers;

    std::vector<std::vector<float>> upsampled_input_buffers;
    std::vector<std::vector<float>> upsampled_output_buffers;

    inline int next_buffer()
    {
      return current_buffer % buffers.size();
    }
#if 0
  
    inline void tanh_activation(int const layer, int const nframes, float const bias)
    {
      std::vector<float> & in_buffer = buffers[current_buffer];

      std::vector<float> & upsampled_input_buffer = upsampled_input_buffers[layer];

      upsamplers[layer].process_block(upsampled_input_buffer.data(), in_buffer.data(), nframes);

      for (int index = 0; index < (2 * nframes); ++index)
      {
        upsampled_input_buffer[index] = tanhf(upsampled_input_buffer[index] + bias);
      }

      downsamplers[layer].process_block(in_buffer.data(), upsampled_input_buffer.data(), nframes);
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
          dist_aa_buffers[layer] = upsampled_input_buffer[index];
        }

        const float x0 = upsampled_input_buffer[index] + bias;
        float x1 = 0;
        if (index == 0)
        {
          x1 = dist_aa_buffers[layer] + bias;
        }
        else
        {
          x1 = upsampled_input_buffer[index - 1] + bias;
        }

        float const x0_2 = x0 * x0;
        float const x1_2 = x1 * x1;

        upsampled_output_buffer[index] = (x0 + x1) / (sqrtf(1 + x0_2) + sqrtf(1 + x1_2));
      }

      downsamplers[layer].process_block(in_buffer.data(), upsampled_output_buffer.data(), nframes);
    }
#endif

    inline void dist_aa2_activation(int const layer, int const nframes, float const bias)
    {
      std::vector<float> & in_buffer = buffers[current_buffer];

      std::vector<float> & upsampled_input_buffer = upsampled_input_buffers[layer];
      std::vector<float> & upsampled_output_buffer = upsampled_output_buffers[layer];

      oversamplers[layer].upsample(in_buffer.data(), upsampled_input_buffer.data(), nframes);

      for (int index = 0; index < (2 * nframes); ++index)
      {
        if (index == (2 * nframes) - 1)
        {
          dist_aa2_buffers[layer][0] = upsampled_input_buffer[index-1];
          dist_aa2_buffers[layer][1] = upsampled_input_buffer[index];
        }

        float x1 = 0;
        float x2 = 0;

        if (index == 0)
        {
          x2 = dist_aa2_buffers[layer][0] + bias;
          x1 = dist_aa2_buffers[layer][1] + bias;
        }
        else if (index == 1)
        {
          x2 = dist_aa2_buffers[layer][1] + bias;
          x1 = upsampled_input_buffer[index - 1] + bias;
        }
        else
        {
          x1 = upsampled_input_buffer[index - 1] + bias;
          x2 = upsampled_input_buffer[index - 2] + bias;
        }
        
        const float x0 = upsampled_input_buffer[index] + bias;

        float const F12 = sqrtf(1.f + ((x0 + x1) / 2.f) * ((x0 + x1) / 2.f));
        float const F1 = sqrtf(1.f + (x1 * x1));
        float const F32 = sqrtf(1.f + ((x1 + x2) / 2.f) * ((x1 + x2) / 2.f));

        upsampled_output_buffer[index] = 0.25f * (((x0 + 3.f*x1) / (F12 + F1)) + ((3.f*x1 + x2) / (F1 + F32)));
      }

      oversamplers[layer].downsample(upsampled_output_buffer.data(), in_buffer.data(), nframes);
    }

    inline void process_layer(int const layer, int const nframes)
    {
      convolvers[layer].process(buffers[current_buffer].data(), buffers[next_buffer()].data(), nframes);
      current_buffer = next_buffer();

      if (activations[layer] == "tanh")
      {
        // tanh_activation(layer, nframes, biases[layer]); 
      }
      else if (activations[layer] == "dist_aa")
      {
        // dist_aa_activation(layer, nframes, biases[layer]); 
      }
      else if (activations[layer] == "dist_aa2")
      {
        dist_aa2_activation(layer, nframes, biases[layer]); 
      }
      else
      {
        std::vector<float> & out_buffer = buffers[current_buffer];
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
      dist_aa_buffers(m.layers.size(), 0),
      dist_aa2_buffers(m.layers.size(), {0, 0}),
      oversamplers(m.layers.size()),
      upsampled_input_buffers(m.layers.size()),
      upsampled_output_buffers(m.layers.size())
    {
      std::cout << "processor()...\n";
  
      for (size_t index = 0; index < m.layers.size(); ++index)
      {
        std::cout << "layer: " << index << "\n";
        convolvers[index].init(blocksize, m.layers[index].weights.data(), m.layers[index].weights.size());
        biases[index] = m.layers[index].bias;
        activations[index] = m.layers[index].activation;
        upsampled_input_buffers[index] = std::vector<float>(2*blocksize);
        upsampled_output_buffers[index] = std::vector<float>(2*blocksize);
      } 

      std::cout << "done.\n";
    }
  };
} 
