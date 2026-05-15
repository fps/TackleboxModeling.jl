#pragma once

#include <FFTConvolver.h>
#include <cmath>
#include <iostream>
#include <array>

namespace tacklebox
{
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
    std::vector<float> buffer1;
    std::vector<float> buffer2;
  
    float x_scale;
    float x_mean;
    float y_scale;
    float y_mean;

    std::vector<fftconvolver::FFTConvolver> convolvers;
    std::vector<float> biases;  
  
    void process(float const * const in, float * const out, float const pre_coef, float const post_coef, int const nframes)
    {
      /*
      for (int index = 0; index < nframes; ++index)
      {
        buffer1[index] = (pre_coef * in[index] - x_mean) / x_scale;
      }
      c1.process(buffer1.data(), buffer2.data(), nframes);
      for (int index = 0; index < nframes; ++index)
      {
        buffer2[index] = tanhf(buffer2[index] + b1);
      }
      c2.process(buffer2.data(), buffer1.data(), nframes);
      for (int index = 0; index < nframes; ++index)
      {
        buffer1[index] = tanhf(buffer1[index] + b2);
      }
      c3.process(buffer1.data(), out, nframes);
      for (int index = 0; index < nframes; ++index)
      {
        out[index] = post_coef * (((out[index] + b3) * y_scale) + y_mean);
      }
      */
    }
 
    /* 
    processor(float const * const w1, const float b1, const int nframes1, float const * const w2, float const b2, int const nframes2, float const * const w3, float const b3, int const nframes3, float const x_scale, float const x_mean, float const y_scale, float const y_mean, int const blocksize) :
      buffer1(blocksize),
      buffer2(blocksize),
      x_scale(x_scale),
      x_mean(x_mean),
      y_scale(y_scale),
      y_mean(y_mean),
      b1(b1),
      b2(b2),
      b3(b3)
    {
      std::cout << "processor()" << std::endl;
      c1.init(blocksize, w1, nframes1);
      std::cout << "1" << std::endl;
      c2.init(blocksize, w2, nframes2);
      std::cout << "2" << std::endl;
      c3.init(blocksize, w3, nframes3);
      std::cout << "processor() done" << std::endl;
    }
    */

    processor(model const & m, int blocksize) :
      buffer1(blocksize),
      buffer2(blocksize),
      x_scale(m.x_scale),
      x_mean(m.x_mean),
      y_scale(m.y_scale),
      y_mean(m.y_mean),
      convolvers(m.layers.size()),
      biases(m.layers.size())
    {
      
    }
  };
} 
