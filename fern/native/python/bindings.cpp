/**
 * Pybind11 bindings for RVQ CUDA kernel
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include "../cuda/rvq_cuda.h"

namespace py = pybind11;

// Helper to convert torch tensor to raw pointer
template<typename T>
T* get_ptr(torch::Tensor& tensor) {
    return tensor.data_ptr<T>();
}

/**
 * Python-facing RVQ decode function
 */
torch::Tensor decode_rvq(
    torch::Tensor codes,        // [B, C, T] int32
    torch::Tensor embeddings    // [C, V, D] float32
) {
    // Validate inputs
    TORCH_CHECK(codes.dtype() == torch::kInt32, "codes must be int32");
    TORCH_CHECK(embeddings.dtype() == torch::kFloat32, "embeddings must be float32");
    TORCH_CHECK(codes.device().is_cuda(), "codes must be on CUDA");
    TORCH_CHECK(embeddings.device().is_cuda(), "embeddings must be on CUDA");
    TORCH_CHECK(codes.dim() == 3, "codes must be 3D [B, C, T]");
    TORCH_CHECK(embeddings.dim() == 3, "embeddings must be 3D [C, V, D]");
    
    // Get dimensions
    int batch_size = codes.size(0);
    int num_codebooks = codes.size(1);
    int time_steps = codes.size(2);
    int vocab_size = embeddings.size(1);
    int embedding_dim = embeddings.size(2);
    
    TORCH_CHECK(embeddings.size(0) == num_codebooks, 
                "embeddings codebooks must match codes codebooks");
    
    // Create output tensor
    auto output = torch::zeros(
        {batch_size, time_steps, embedding_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(codes.device())
    );
    
    // Create config
    ilava::cuda::RVQConfig config(
        batch_size, num_codebooks, time_steps, 
        embedding_dim, vocab_size
    );
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call CUDA kernel
    cudaError_t err = ilava::cuda::decode_rvq_cuda(
        get_ptr<int32_t>(codes),
        get_ptr<float>(embeddings),
        get_ptr<float>(output),
        config,
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess, 
                "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

/**
 * Python-facing amortized RVQ decode
 */
torch::Tensor decode_rvq_amortized(
    torch::Tensor codes,
    torch::Tensor embeddings,
    torch::Tensor frame_mask    // [T] bool
) {
    // Validate
    TORCH_CHECK(frame_mask.dtype() == torch::kBool, "frame_mask must be bool");
    TORCH_CHECK(frame_mask.device().is_cuda(), "frame_mask must be on CUDA");
    TORCH_CHECK(frame_mask.dim() == 1, "frame_mask must be 1D");
    TORCH_CHECK(frame_mask.size(0) == codes.size(2), 
                "frame_mask length must match time_steps");
    
    // Get dimensions
    int batch_size = codes.size(0);
    int num_codebooks = codes.size(1);
    int time_steps = codes.size(2);
    int vocab_size = embeddings.size(1);
    int embedding_dim = embeddings.size(2);
    
    // Create output
    auto output = torch::zeros(
        {batch_size, time_steps, embedding_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(codes.device())
    );
    
    // Config
    ilava::cuda::RVQConfig config(
        batch_size, num_codebooks, time_steps,
        embedding_dim, vocab_size
    );
    
    // Get stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call kernel
    cudaError_t err = ilava::cuda::decode_rvq_cuda_amortized(
        get_ptr<int32_t>(codes),
        get_ptr<float>(embeddings),
        get_ptr<float>(output),
        config,
        get_ptr<bool>(frame_mask),
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

// Module definition
PYBIND11_MODULE(rvq_cuda_ext, m) {
    m.doc() = "RVQ CUDA kernel for i-LAVA";
    
    m.def("decode_rvq", &decode_rvq,
          "Decode RVQ codes to embeddings using CUDA",
          py::arg("codes"),
          py::arg("embeddings"));
    
    m.def("decode_rvq_amortized", &decode_rvq_amortized,
          "Decode RVQ codes with compute amortization (1/16 sampling)",
          py::arg("codes"),
          py::arg("embeddings"),
          py::arg("frame_mask"));
}

