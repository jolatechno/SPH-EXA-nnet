#pragma once

#include "../eigen.hpp"

#ifdef USE_CUDA
	#include <cuda_runtime.h>
#endif

// base implementations
namespace eigen::cuda {
	/// custom fixed-size matrix type
	template<typename Type, int n, int m>
	class fixed_size_matrix {
	private:
		Type *weights;

	public:
		__host__ fixed_size_matrix() {
			cudaMalloc(&weights, n*m*sizeof(Type));
		}
		__host__ fixed_size_matrix(const eigen::fixed_size_matrix<Type, n, m> &other) : fixed_size_matrix() {
			cudaMemcpy(weights, other.data(), n*m*sizeof(Type), cudaMemcpyHostToDevice);
		}
		__host__ ~fixed_size_matrix() {
			cudaFree(weights);
		}

		__device__ Type inline &operator()(int i, int j) {
			return weights[i + j*n];
		}
		__device__ Type inline operator()(int i, int j) const {
			return weights[i + j*n];
		}
		const Type *data() const {
			return weights;
		}
	};

	/// custom fixed sizes array
	template<typename Type, int n>
	class array {
	private:
		Type *weights;

	public:
		__host__ array() {
			cudaMalloc(&weights, n*sizeof(Type));
		}
		__host__ array(const std::array<double, n> &other) : array() {
			cudaMemcpy(weights, other.data(), n*sizeof(Type), cudaMemcpyHostToDevice);
		}
		__host__ ~array() {
			cudaFree(weights);
		}

		__device__ Type inline &operator[](int i) {
			return weights[i];
		}

		__device__ Type inline operator[](int i) const {
			return weights[i];
		}
	};
}