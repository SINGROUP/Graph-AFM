
// Pytorch C++/Cuda extension tutorial:
// https://pytorch.org/tutorials/advanced/cpp_extension.html

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK3D(x) TORCH_CHECK(x.dim() == 3, #x " must be a 3D tensor")
#define CHECK4D(x) TORCH_CHECK(x.dim() == 4, #x " must be a 4D tensor")

#define MAX_LABELS 1024

enum TMMethod {
    TM_MAD,
    TM_MSD,
    TM_MAD_NORM,
    TM_MSD_NORM
};

torch::Tensor match_template_cuda(torch::Tensor array, torch::Tensor temp, TMMethod method);
torch::Tensor ccl_cuda(torch::Tensor array);
std::vector<torch::Tensor> find_label_min_cuda(torch::Tensor array, torch::Tensor labels);
