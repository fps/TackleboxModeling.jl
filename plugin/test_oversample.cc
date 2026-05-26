#include <oversample.h>
#include <iostream>
#include <cmath>
#include <vector>

int main()
{
  oversample2x o;

  std::vector<float> input(48000, 0);
  std::vector<float> oversampled(96000, 0);
  std::vector<float> output(48000, 0);

  for (int n = 0; n < 48000; ++n)
  {
    input[n] = sin(2 * M_PI * 0.01 * n);
  }

  o.upsample(input.data(), oversampled.data(), 48000);
  for (int n = 0; n < 96000; ++n)
  {
    oversampled[n] = tanh(oversampled[n]);
  }
  o.downsample(oversampled.data(), output.data(), 48000);

  for (int n = 0; n < 48000; ++n)
  {
    std::cout << output[n] << "\n";
  }
}

