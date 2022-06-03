#include <torch/script.h>
// #include <torchaudio/csrc/rnnt/cpu/cpu_transducer.h>

namespace torchaudio {
namespace rnnt {
namespace cpu {
std::tuple<torch::Tensor, c10::optional<torch::Tensor>> compute(
    torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& logit_lengths,
    const torch::Tensor& target_lengths,
    int64_t blank,
    double clamp) {
  std::cout << "in cpu dummpy compute" << std::endl;
  c10::optional<torch::Tensor> gradients = torch::zeros_like(logits);
  return std::make_tuple(logits, gradients);
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss", &compute);
}

} // namespace cpu
} // namespace rnnt
} // namespace torchaudio
