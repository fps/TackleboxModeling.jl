#pragma once

#include <FFTConvolver.h>
#include <cmath>
#include <iostream>

struct tacklebox
{
  std::vector<float> m_buffer1;
  std::vector<float> m_buffer2;

  float m_x_scale;
  float m_x_mean;
  float m_y_scale;
  float m_y_mean;

  fftconvolver::FFTConvolver c1;
  fftconvolver::FFTConvolver c2;
  fftconvolver::FFTConvolver c3;

  float m_b1;
  float m_b2;
  float m_b3;

  void process(float const * const in, float * const out, int nframes)
  {
    for (int index = 0; index < nframes; ++index)
    {
      m_buffer1[index] = (in[index] - m_x_mean) / m_x_scale;
    }
    c1.process(m_buffer1.data(), m_buffer2.data(), nframes);
    for (int index = 0; index < nframes; ++index)
    {
      m_buffer2[index] = tanhf(m_buffer2[index] + m_b1);
    }
    c2.process(m_buffer2.data(), m_buffer1.data(), nframes);
    for (int index = 0; index < nframes; ++index)
    {
      m_buffer1[index] = tanhf(m_buffer1[index] + m_b2);
    }
    c3.process(m_buffer1.data(), out, nframes);
    for (int index = 0; index < nframes; ++index)
    {
      out[index] = ((out[index] + m_b3) * m_y_scale) + m_y_mean;
    }
  }

  tacklebox(float const * const w1, const float b1, const int nframes1, float const * const w2, float const b2, int const nframes2, float const * const w3, float const b3, int const nframes3, float const x_scale, float const x_mean, float const y_scale, float const y_mean, int const blocksize) :
    m_buffer1(blocksize),
    m_buffer2(blocksize),
    m_x_scale(x_scale),
    m_x_mean(x_mean),
    m_y_scale(y_scale),
    m_y_mean(y_mean),
    m_b1(b1),
    m_b2(b2),
    m_b3(b3)
  {
    std::cout << "tacklebox()" << std::endl;
    c1.init(blocksize, w1, nframes1);
    std::cout << "1" << std::endl;
    c2.init(blocksize, w2, nframes2);
    std::cout << "2" << std::endl;
    c3.init(blocksize, w3, nframes3);
    std::cout << "tacklebox() done" << std::endl;
  }
};

