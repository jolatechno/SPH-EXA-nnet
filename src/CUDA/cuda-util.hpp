#pragma once

#include <cuda_runtime.h>


__host__ void gpuErrchk(cudaError_t code) {
	if (code != cudaSuccess) {
#ifdef CUDA_ERROR_FATAL
		std::string err = "CUDA error (fatal) ! \"";
		err += cudaGetErrorString(code);
		err += "\"\n";
		
		throw std::runtime_error(err);
#else
		std::cerr << "\tCUDA error (non-fatal) ! \"" << cudaGetErrorString(code) << "\"\n";
#endif
	}
}

namespace cuda_util {
	/// function to move a buffer to the GPU
	/**
	 * TODO
	 */
	template<class T>
	T *move_to_gpu(const T* const ptr, int dimension) {
		T *dev_ptr;
		
		gpuErrchk(cudaMalloc((void**)&dev_ptr, dimension*sizeof(T)));
		gpuErrchk(cudaMemcpy(dev_ptr,     ptr, dimension*sizeof(T), cudaMemcpyHostToDevice));

		return dev_ptr;
	}

	/// function to move a cpu buffer to a gpu symbol
	template<class T>
	void move_to_gpu(T *&dev_ptr, const T* host_values, size_t N) {
		T *ptr;
		gpuErrchk(cudaMalloc((void **)&ptr, N*sizeof(T)));
		gpuErrchk(cudaMemcpy(ptr, host_values, N*sizeof(T), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy((void **)&dev_ptr, (void **)&ptr, sizeof(T*), cudaMemcpyHostToDevice));
	}
}
