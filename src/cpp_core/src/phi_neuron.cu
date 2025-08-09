#include "redcode/phi_neuron.h"
#include <cuda_runtime.h>
#include <iostream>

namespace redcode {

// CUDA kernel to apply the PhiNeuron forward pass to an array of floats.
// Each thread in the grid processes one element of the array.
__global__ void phi_neuron_forward_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        PhiNeuron neuron; // Create an instance of the neuron on the device.
        // The forward method is marked __device__, so it can be called from the kernel.
        data[idx] = neuron.forward(data[idx]);
    }
}

// C++ wrapper function to launch the CUDA kernel.
// This function orchestrates the entire host-side process for a GPU computation.
void apply_phi_neuron_forward_gpu(float* host_data, size_t n) {
    // 1. Allocate memory on the GPU device
    float* device_data;
    cudaError_t err = cudaMalloc(&device_data, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "GPU Error: Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 2. Copy input data from host (CPU) to device (GPU)
    err = cudaMemcpy(device_data, host_data, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "GPU Error: Failed to copy data to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_data);
        return;
    }

    // 3. Define kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 4. Launch the kernel on the device
    phi_neuron_forward_kernel<<<blocks_per_grid, threads_per_block>>>(device_data, n);

    // Check for any errors launched by the kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "GPU Error: Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_data);
        return;
    }

    // 5. Copy results back from device to host
    err = cudaMemcpy(host_data, device_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "GPU Error: Failed to copy data from device: " << cudaGetErrorString(err) << std::endl;
    }

    // 6. Free GPU memory
    cudaFree(device_data);
}

} // namespace redcode
