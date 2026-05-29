#include <tacklebox.h>
#include <vector>
#include <iostream>

int main()
{
  std::cout << "Initializing...\n";

  std::vector<tacklebox::model> ms = { 
    #include "../data/EVH 5150/model.cc"
  };

  tacklebox::processor t(ms[0], 64);

  std::vector<float> in(64);
  std::vector<float> out(64);

  std::cout << "Processing)...\n";

  for (int index = 0; index < (10 * 750); ++index)
  {
    t.process(in.data(), out.data(), 1.f, 1.f, 1, 0, 64);
  }
}
