#include <tacklebox.h>
#include <vector>

int main()
{
  std::vector<float> w1(256);
  std::vector<float> w2(512);
  std::vector<float> w3(1024);
  tacklebox t(w1.data(), 1.f, w1.size(), w2.data(), 1.f, w2.size(), w3.data(), 1.f, w3.size(), 1.f, 0.f, 1.f, 0.f, 64);

  std::vector<float> in(64);
  std::vector<float> out(64);

  for (int index = 0; index < (1000 * 750); ++index)
  {
    t.process(in.data(), out.data(), 64);
  }
}
