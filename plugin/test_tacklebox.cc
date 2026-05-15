#include <tacklebox.h>
#include <vector>

int main()
{
  tacklebox::params<1, 1, 1> p = { { 1.f }, { 1.f}, { 1.f }, 1.f, 1.f, 1.f, 1.f, 0.f, 1.f, 0.f };

  std::vector<float> w1(256, 1.1f);
  std::vector<float> w2(512, 1.2f);
  std::vector<float> w3(1024, 1.3f);
  tacklebox::model t(w1.data(), 1.f, w1.size(), w2.data(), 1.f, w2.size(), w3.data(), 1.f, w3.size(), 1.f, 0.f, 1.f, 0.f, 64);

  std::vector<float> in(64);
  std::vector<float> out(64);

  for (int index = 0; index < (1000 * 750); ++index)
  {
    t.process(in.data(), out.data(), 64);
  }
}
