#include <tacklebox.h>
#include <vector>

int main()
{
  std::vector<tacklebox::model> ms = { 
    #include "../data/BrianMay/model.cc"
  };

  tacklebox::processor t(ms[0], 64);

  std::vector<float> in(64);
  std::vector<float> out(64);

  for (int index = 0; index < (1000 * 750); ++index)
  {
    t.process(in.data(), out.data(), 1.f, 1.f, 64);
  }
}
