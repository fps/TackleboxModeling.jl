#include <oversample.h>
#include <iostream>
#include <cmath>
#include <vector>


const int T = 10;
const int RATE = 48000;
const int L = RATE*T;

int main()
{
  oversample2x o;

  std::vector<float> input(L, 0);
  std::vector<float> oversampled(L*2, 0);
  std::vector<float> output(L, 0);

  float phase = 0;
  for (int n = 0; n < L; ++n)
  {
    input[n] = sin(phase);
    phase += 2 * M_PI * 0.5 * (float)n/(float)L;
    if (phase > 2 * M_PI)
    {
      phase -= 2 * M_PI;
    }
  }

  o.upsample(input.data(), oversampled.data(), L);
  for (int n = 0; n < (2*L); ++n)
  {
    // oversampled[n] = tanh(100 * oversampled[n]);
    oversampled[n] = tanh(100.f * oversampled[n]);
  }
  o.downsample(oversampled.data(), output.data(), L);
 
  for (int n = 0; n < L; ++n)
  {
    std::cout << input[n] << " " << output[n] << "\n";
  }
}

