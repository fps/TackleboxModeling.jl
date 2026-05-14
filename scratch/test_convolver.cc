#include <FFTConvolver.h>
#include <vector>

int main()
{
  std::vector<float> weights(1024);
  fftconvolver::FFTConvolver c;
  c.init(64, weights.data(), 1024);

  std::vector<float> in(64);
  std::vector<float> out(64);

  for (int n = 0; n < (10000 * 4096); ++n)
  {
    c.process(in.data(), out.data(), 64);
  }
}
