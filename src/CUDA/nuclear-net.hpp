#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "../nuclear-net.hpp"

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
}


namespace nnet {
	/// class for reactions on gpu
	/**
	 * TODO
	 */
	class gpu_reaction_list : public ptr_reaction_list {
		friend gpu_reaction_list move_to_gpu(const ptr_reaction_list &reactions);

	public:
		gpu_reaction_list() {}
		~gpu_reaction_list() {
			gpuErrchk(cudaFree((void*)ptr_reaction_list::reactant_product));
			gpuErrchk(cudaFree((void*)ptr_reaction_list::reactant_begin));
			gpuErrchk(cudaFree((void*)ptr_reaction_list::product_begin));
		}
	};


	/// function to move reactions to the GPU
	/**
	 * TODO
	 */
	gpu_reaction_list move_to_gpu(const ptr_reaction_list &reactions) {
		gpu_reaction_list dev_reactions;
		dev_reactions.num_reactions = reactions.num_reactions;

		dev_reactions.reactant_product = cuda_util::move_to_gpu<reaction::reactant_product>(reactions.reactant_product, reactions.reactant_begin[reactions.num_reactions]);
		dev_reactions.reactant_begin   = cuda_util::move_to_gpu<int>(reactions.reactant_begin, reactions.num_reactions + 1);
		dev_reactions.product_begin    = cuda_util::move_to_gpu<int>(reactions.product_begin,  reactions.num_reactions);

		return dev_reactions;
	} 
}