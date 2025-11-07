/**
 * RVQ CUDA Kernel for i-LAVA
 * 
 * Optimized CUDA implementation of Residual Vector Quantization decoding.
 * 
 * Performance targets:
 * - 10x faster than PyTorch implementation
 * - Efficient shared memory usage
 * - Coalesced memory access
 * 
 * Author: i-LAVA Project
 * Date: November 2025
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ilava {
namespace cuda {

/**
 * RVQ Decoding Configuration
 */
struct RVQConfig {
    int batch_size;        // Batch size
    int num_codebooks;     // Number of RVQ codebooks
    int time_steps;        // Number of time steps
    int embedding_dim;     // Embedding dimension
    int vocab_size;        // Vocabulary size per codebook
    
    RVQConfig(int bs, int nc, int ts, int ed, int vs)
        : batch_size(bs), num_codebooks(nc), time_steps(ts),
          embedding_dim(ed), vocab_size(vs) {}
};

/**
 * Decode RVQ codes to embeddings using CUDA.
 * 
 * Performs parallel codebook lookups and accumulation:
 * For each (batch, time_step):
 *     output[b,t,:] = sum(embeddings[c, codes[b,c,t], :] for c in codebooks)
 * 
 * @param codes        Input codes [batch, num_codebooks, time_steps] (int32)
 * @param embeddings   Embedding tables [num_codebooks, vocab_size, embedding_dim] (float32)
 * @param output       Output embeddings [batch, time_steps, embedding_dim] (float32)
 * @param config       RVQ configuration
 * @param stream       CUDA stream (optional, default stream if nullptr)
 * 
 * @return cudaError_t Error code
 */
cudaError_t decode_rvq_cuda(
    const int32_t* codes,
    const float* embeddings,
    float* output,
    const RVQConfig& config,
    cudaStream_t stream = nullptr
);

/**
 * Decode RVQ codes with compute amortization (1/16 sampling).
 * 
 * Only processes a subset of frames for efficiency during training/inference.
 * 
 * @param codes        Input codes [batch, num_codebooks, time_steps]
 * @param embeddings   Embedding tables
 * @param output       Output embeddings [batch, time_steps, embedding_dim]
 * @param config       RVQ configuration
 * @param frame_mask   Boolean mask [time_steps] indicating which frames to process
 * @param stream       CUDA stream
 * 
 * @return cudaError_t Error code
 */
cudaError_t decode_rvq_cuda_amortized(
    const int32_t* codes,
    const float* embeddings,
    float* output,
    const RVQConfig& config,
    const bool* frame_mask,
    cudaStream_t stream = nullptr
);

/**
 * Get optimal grid and block dimensions for RVQ kernel.
 * 
 * @param config RVQ configuration
 * @param grid   Output grid dimensions
 * @param block  Output block dimensions
 */
void get_rvq_launch_config(
    const RVQConfig& config,
    dim3& grid,
    dim3& block
);

} // namespace cuda
} // namespace ilava

