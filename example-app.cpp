#include <torch/torch.h>
#include <iostream>

#include<rnnt/macros.h>

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

  return 0;
}
