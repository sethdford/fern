# Native Optimizations for i-LAVA

This directory contains native C++/CUDA implementations for performance-critical operations.

## Structure

```
native/
├── cuda/                   # CUDA kernels
│   ├── rvq_cuda.cu        # RVQ decoding kernel
│   ├── rvq_cuda.h         # Header file
│   └── tests/             # CUDA tests
├── python/                 # Python bindings
│   ├── __init__.py
│   ├── rvq_cuda.py        # RVQ wrapper
│   └── bindings.cpp       # pybind11 bindings
├── tests/                  # Integration tests
│   └── test_rvq_cuda.py
├── CMakeLists.txt         # Build configuration
└── setup.py               # Python package setup
```

## Requirements

### For CUDA Implementation
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- CMake 3.18+
- pybind11

### For Building
```bash
pip install pybind11
# Install CUDA Toolkit from NVIDIA

# Build
mkdir build && cd build
cmake ..
make -j4

# Install
pip install -e .
```

## Features

### Implemented
- [x] Project structure
- [ ] RVQ CUDA kernel
- [ ] Python bindings
- [ ] Tests
- [ ] Benchmarks

### Performance Targets
- RVQ Decoding: 10x speedup vs PyTorch
- Memory: Efficient shared memory usage
- Latency: < 5ms for typical inputs

## Usage

```python
from fern.native import rvq_cuda_available, decode_rvq_cuda

if rvq_cuda_available():
    # Use CUDA implementation
    output = decode_rvq_cuda(codes, embeddings)
else:
    # Fallback to PyTorch
    output = decode_rvq_pytorch(codes, embeddings)
```

## Development

See `docs/CUDA_DEVELOPMENT.md` for detailed development guide.

