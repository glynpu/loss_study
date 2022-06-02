#include <torch/torch.h>
#include <iostream>

#include<rnnt/macros.h>
#include<rnnt/types.h>
#include<rnnt/options.h>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  std::cout << WARP_SIZE << std::endl;
  // std::cout << FORCE_INLINE << std::endl;
  level_t x = INFO;
  std::cout << x << std::endl;
  x = ERROR;
  std::cout << x << std::endl;
  std::cout << ToString(x) << std::endl;

  torchaudio::rnnt::status_t y = torchaudio::rnnt::status_t::SUCCESS;
  std::cout << torchaudio::rnnt::toString(y) << std::endl;

  torchaudio::rnnt::Options o;
  std::cout << o << std::endl;

  return 0;
}
