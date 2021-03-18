#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <c10/cuda/CUDAStream.h>

// namespace {
__global__ void SCAN_NN_Mask_Fill_cuda_kernel(
    torch::PackedTensorAccessor<int64_t,1,torch::RestrictPtrTraits,size_t> p_idx,
    torch::PackedTensorAccessor<int64_t,2,torch::RestrictPtrTraits,size_t> nearest_neighbors,
    torch::PackedTensorAccessor<int32_t,2,torch::RestrictPtrTraits,size_t> output_mask) {
    // //batch index
    // const int row_idx = blockIdx.y;
    // // column index
    // const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t neighbor_idx = blockIdx.z;
    if(row_idx >= output_mask.size(0) || col_idx >= output_mask.size(1) || col_idx >= p_idx.size(0) || neighbor_idx >= nearest_neighbors.size(1) || row_idx >= nearest_neighbors.size(0)) {return;}
    // int32_t isneighbor = (p_idx[col_idx] == nearest_neighbors[row_idx][neighbor_idx]) and !(col_idx == row_idx);
    int32_t isneighbor = (p_idx[col_idx] == nearest_neighbors[row_idx][neighbor_idx]);
    //  and !(col_idx == row_idx);
    atomicOr(&output_mask[row_idx][col_idx], isneighbor);
}
// } // namespace


void SCAN_NN_Mask_Fill_cuda(
    torch::Tensor p_idx,
    torch::Tensor nearest_neighbors,
    torch::Tensor output_mask)
{
    // const int threads = 1024;
    const auto b_size = p_idx.size(0);
    const auto n_nearest = nearest_neighbors.size(1);
    const dim3 threadsPerBlock(32, 32, 1);
    const dim3 blocks((b_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (b_size + threadsPerBlock.y - 1) / threadsPerBlock.y, n_nearest);
    // assert(p_idx.size(0) == n_nearest.size(0), "p_idx and nearest size at dim 0 must be equal.");
    // assert(p_idx.size(0) == output_mask.size(0), "output mask should match dimensions of p_idx");
    // assert(p_idx.size(0) == output_mask.size(1), "output mask should match dimensions of p_idx");

    // std::cout << "test " << p_idx.size(0) << std::endl;
    // std::cout << "test " << nearest_neighbors.size(0) << std::endl;
    assert(p_idx.size(0) == nearest_neighbors.size(0));
    assert(p_idx.size(0) == output_mask.size(0));
    assert(p_idx.size(0) == output_mask.size(1));

    auto stream = at::cuda::getCurrentCUDAStream(p_idx.device().index());

    SCAN_NN_Mask_Fill_cuda_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
        p_idx.packed_accessor<int64_t,1,torch::RestrictPtrTraits,size_t>(),
        nearest_neighbors.packed_accessor<int64_t,2,torch::RestrictPtrTraits,size_t>(),
        output_mask.packed_accessor<int32_t,2,torch::RestrictPtrTraits,size_t>());
}
