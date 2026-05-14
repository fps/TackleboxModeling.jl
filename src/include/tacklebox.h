#pragma once

#include <FFTConvolver.h>
#include <cmath>

struct tacklebox
{
  fftconvolver::FFTConvolver c1;
  fftconvolver::FFTConvolver c2;
  fftconvolver::FFTConvolver c3;

  float m_x_scale;
  float m_x_mean;
  float m_y_scale;
  float m_y_mean;

  std::vector<float> m_buffer1;
  std::vector<float> m_buffer2;

  void process(float const *in, float *out, int nframes)
  {
    c1.process(in, m_buffer1.data(), nframes);
    for (int index = 0; index < nframes; ++index)
    {
      m_buffer1[index] = tanhf(m_buffer1[index]);
    }
    c2.process(m_buffer1.data(), m_buffer2.data(), nframes);
    for (int index = 0; index < nframes; ++index)
    {
      m_buffer2[index] = tanhf(m_buffer2[index]);
    }
    c3.process(m_buffer2.data(), out, nframes);
  }

  tacklebox(float *w1, int nframes1, float *w2, int nframes2, float *w3, int nframes3, float x_scale, float x_mean, float y_scale, float, y_mean, int blocksize) :
    m_buffer1(blocksize),
    m_buffer2(blocksize),
    m_x_scale(x_scale),
    m_x_mean(x_mean),
    m_y_scale(y_scale),
    m_y_mean(y_mean)
  {
    c1.init(blocksize, w1, nframes1);
    c2.init(blocksize, w2, nframes2);
    c3.init(blocksize, w3, nframes3);
  }
};

