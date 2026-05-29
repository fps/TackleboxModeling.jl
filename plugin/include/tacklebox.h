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

    enum OVERSAMPLING {
      DISABLED = 0,
      OVERSAMPLE2X,
    };

    std::vector<oversample2x> oversamplers;

    std::vector<float> upsampled_input_buffer;
    std::vector<float> upsampled_output_buffer;

    inline int next_buffer()
    {
      return (current_buffer + 1) % 2;
    }

    inline void dist_aa2_activation(int const layer, float *input, float *output, int const nframes)
    {
      for (int index = 0; index < nframes; ++index)
      {
        float const x2 = dist_aa2_buffers[layer][0];
        float const x1 = dist_aa2_buffers[layer][1];
        float const x0 = input[index];

        float const x01_2 = ((x0 + x1) / 2.f);
        float const x01_s = x01_2 * x01_2;

        float const x12_2 = ((x1 + x2) / 2.f);
        float const x12_s = x12_2 * x12_2;

        float const F12 = sqrtf(1.f + x01_s);
        float const F1 = sqrtf(1.f + (x1 * x1));
        float const F32 = sqrtf(1.f + x12_s);

        float const x1_3 = 3.f * x1;

        output[index] = 0.25f * (((x0 + x1_3) / (F12 + F1)) + ((x1_3 + x2) / (F1 + F32)));

        dist_aa2_buffers[layer][0] = x1;
        dist_aa2_buffers[layer][1] = x0;
      }
    }

    inline void process_layer(int const layer, int oversampling, int const nframes)
    {
      convolvers[layer].process(buffers[current_buffer].data(), buffers[next_buffer()].data(), nframes);
      current_buffer = next_buffer();

      for (int frame = 0; frame < nframes; ++frame)
      {
        buffers[current_buffer][frame] += biases[layer];
      }

      if (activations[layer] == "nothing")
      {

      } 
      else
      {
        if (oversampling)
        {
          auto & oversampler = oversamplers[layer];
          oversampler.upsample(buffers[current_buffer].data(), upsampled_input_buffer.data(), nframes);
  
          if (activations[layer] == "dist_aa2")
          {
            dist_aa2_activation(layer, upsampled_input_buffer.data(), upsampled_output_buffer.data(), 2*nframes); 
          }
          else
          {
            throw std::runtime_error("Unsupported activation");
          }
  
          oversampler.downsample(upsampled_output_buffer.data(), buffers[current_buffer].data(), nframes);
        }
        else
        {
          if (activations[layer] == "dist_aa2")
          {
            dist_aa2_activation(layer, buffers[current_buffer].data(), buffers[current_buffer].data(), nframes);
          }
          else
          {
            throw std::runtime_error("Unsupported activation");
          }
        }
      }
    }
  
    inline void process(float const * const in, float * const out, float const pre_coef, float const post_coef, int const oversampling, int const stage_to_process, int const nframes)
    {
      // std::cout << pre_coef << " " << post_coef << "\n";
      if (nframes > (int)buffers[0].size())
      {
        throw std::runtime_error("nframes > blocksize");
      }

      current_buffer = 0;
      std::vector<float> & in_buffer = buffers[current_buffer];

       // If processing all stages (stage_to_process == 0) or just the first stage then apply scaling factors...
      if (stage_to_process <= 1)
      {
 
        for (int index = 0; index < nframes; ++index)
        {
          in_buffer[index] = (pre_coef * in[index] - x_mean) / x_scale;
        }
      }
      // Otherwise just copy the input into the input buffer..
      else
      {
        for (int index = 0; index < nframes; ++index)
        {
          in_buffer[index] = in[index];
        }
      }

      // If stage_to_process is 0, process the whole thing...
      if (stage_to_process == 0)
      {
        for (size_t layer = 0; layer < biases.size(); ++layer)
        {
          process_layer(layer, oversampling, nframes);
        }
      } 
      // Otherwise just process the requested stage
      else
      {
        process_layer(stage_to_process - 1, oversampling, nframes);
      }

      std::vector<float> &out_buffer = buffers[current_buffer];

      // If we are processing the whole thing or just the last stage apply scaling factors
      if (stage_to_process == 0 || stage_to_process == (int)biases.size() - 1)     
      {
        for (int index = 0; index < nframes; ++index)
        {
          out[index] = post_coef * ((out_buffer[index] * y_scale) + y_mean);
        }
      }
      else
      {
        for (int index = 0; index < nframes; ++index)
        {
          out[index] = out_buffer[index];
        }
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
      for (size_t index = 0; index < m.layers.size(); ++index)
      {
        convolvers[index].init(blocksize, m.layers[index].weights.data(), m.layers[index].weights.size());
        biases[index] = m.layers[index].bias;
        activations[index] = m.layers[index].activation;
      } 
    }
  };
} 
