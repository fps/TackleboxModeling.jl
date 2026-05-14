#pragma once

#include <FFTConvolver.h>
#include <cmath>

struct tacklebox
{
  fftconvolver::FFTConvolver c1;
  fftconvolver::FFTConvolver c2;
  fftconvolver::FFTConvolver c3;

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

  tacklebox(float *w1, int nframes1, float *w2, int nframes2, float *w3, int nframes3, int blocksize) :
    m_buffer1(blocksize),
    m_buffer2(blocksize)
  {
    c1.init(blocksize, w1, nframes1);
    c2.init(blocksize, w2, nframes2);
    c3.init(blocksize, w3, nframes3);
  }
};

