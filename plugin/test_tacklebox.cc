#include <tacklebox.h>
#include <vector>
#include <iostream>

int main()
{
  std::cout << "Initializing...\n";

  std::vector<tacklebox::model> ms = { 
    #include "../data/BrianMay/model.cc"
  };

  tacklebox::processor t(ms[0], 64);

  std::vector<float> in(64);
  std::vector<float> out(64);

  std::cout << "Processing 1000 secs of audio (1000 * 750 * 64 frames = 48000000 frames @ 48 Khz)...\n";

  for (int index = 0; index < (1000 * 750); ++index)
  {
    t.process(in.data(), out.data(), 1.f, 1.f, 64);
  }
}
