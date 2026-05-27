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

    std::vector<float> upsampled_input_buffer;
    std::vector<float> upsampled_output_buffer;

    inline int next_buffer()
    {
      return (current_buffer + 1) % 2;
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

    inline void dist_aa2_activation(int const layer, int const nframes)
    {
      std::vector<float> & buffer = buffers[current_buffer];

      auto & oversampler = oversamplers[layer];
      oversampler.upsample(buffer.data(), upsampled_input_buffer.data(), nframes);

      for (int index = 0; index < (2 * nframes); ++index)
      {
        float const x2 = dist_aa2_buffers[layer][0];
        float const x1 = dist_aa2_buffers[layer][1];
        float const x0 = upsampled_input_buffer[index];

        float const x01_2 = ((x0 + x1) / 2.f);
        float const x01_s = x01_2 * x01_2;

        float const x12_2 = ((x1 + x2) / 2.f);
        float const x12_s = x12_2 * x12_2;

        float const F12 = sqrtf(1.f + x01_s);
        float const F1 = sqrtf(1.f + (x1 * x1));
        float const F32 = sqrtf(1.f + x12_s);

        float const x1_3 = 3.f * x1;

        upsampled_output_buffer[index] = 0.25f * (((x0 + x1_3) / (F12 + F1)) + ((x1_3 + x2) / (F1 + F32)));

        // upsampled_output_buffer[index] = (x0 + x1) / (sqrtf(1.f + x0 * x0) + sqrtf(1.f + x1 * x1));
        // upsampled_output_buffer[index] = tanhf(100.f *  x0);
        // upsampled_output_buffer[index] = x0;

        dist_aa2_buffers[layer][0] = x1;
        dist_aa2_buffers[layer][1] = x0;
      }

      oversampler.downsample(upsampled_output_buffer.data(), buffer.data(), nframes);
      // oversamplers[layer].downsample(upsampled_input_buffer.data(), buffer.data(), nframes);
    }

    inline void process_layer(int const layer, int const nframes)
    {
      convolvers[layer].process(buffers[current_buffer].data(), buffers[next_buffer()].data(), nframes);
      current_buffer = next_buffer();

      // std::cout << current_buffer << "\n";

      for (int frame = 0; frame < nframes; ++frame)
      {
        buffers[current_buffer][frame] += biases[layer];
      }

      if (activations[layer] == "tanh")
      {
        throw std::runtime_error("Unsupported activation");
        // tanh_activation(layer, nframes, biases[layer]); 
      }
      else if (activations[layer] == "dist_aa")
      {
        throw std::runtime_error("Unsupported activation");
        // dist_aa_activation(layer, nframes, biases[layer]); 
      }
      else if (activations[layer] == "dist_aa2")
      {
        dist_aa2_activation(layer, nframes); 
      }
      else if (activations[layer] == "nothing")
      {

      }
      else
      {
        throw std::runtime_error("Unsupported activation");
      }
    }
  
    inline void process(float const * const in, float * const out, float const pre_coef, float const post_coef, int const nframes)
    {
      // std::cout << pre_coef << " " << post_coef << "\n";
      if (nframes > (int)buffers[0].size())
      {
        throw std::runtime_error("nframes > blocksize");
      }

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
      buffers{std::vector<float>(blocksize, 0), std::vector<float>(blocksize, 0)},
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
      upsampled_input_buffer(2*blocksize),
      upsampled_output_buffer(2*blocksize)
    {
      std::cout << "processor()...\n";
  
      for (size_t index = 0; index < m.layers.size(); ++index)
      {
        std::cout << "layer: " << index << "\n";
        convolvers[index].init(blocksize, m.layers[index].weights.data(), m.layers[index].weights.size());
        biases[index] = m.layers[index].bias;
        activations[index] = m.layers[index].activation;
      } 

      std::cout << "done.\n";
    }
  };
} 
