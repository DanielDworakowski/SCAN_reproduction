#include <torch/extension.h>
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declaration.
void SCAN_NN_Mask_Fill_cuda(
        torch::Tensor p_idx,
        torch::Tensor nearest_neighbors,
        torch::Tensor output_mask);

// CPP interface.
void SCAN_NN_Mask_Fill(
        torch::Tensor p_idx,
        torch::Tensor nearest_neighbors,
        torch::Tensor output_mask)
{
    CHECK_INPUT(p_idx);
    CHECK_INPUT(nearest_neighbors);
    CHECK_INPUT(output_mask);
    return SCAN_NN_Mask_Fill_cuda(p_idx, nearest_neighbors, output_mask);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SCAN_NN_Mask_Fill", &SCAN_NN_Mask_Fill, "SCAN_NN_Mask_Fill");
}
