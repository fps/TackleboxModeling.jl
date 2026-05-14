#pragma once

#include <FFTConvolver.h>
#include <cmath>

struct tacklebox
{
  fftconvolver::FFTConvolver c1;
  fftconvolver::FFTConvolver c2;
  fftconvolver::FFTConvolver c3;

  float m_b1;
  float m_b2;
  float m_b3;

  std::vector<float> m_buffer1;
  std::vector<float> m_buffer2;

  float m_x_scale;
  float m_x_mean;
  float m_y_scale;
  float m_y_mean;

  void process(float const *in, float *out, int nframes)
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

  tacklebox(float *w1, float b1, int nframes1, float *w2, float b2, int nframes2, float *w3, float b3, int nframes3, float x_scale, float x_mean, float y_scale, float y_mean, int blocksize) :
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
    c1.init(blocksize, w1, nframes1);
    c2.init(blocksize, w2, nframes2);
    c3.init(blocksize, w3, nframes3);
  }
};

