#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "../nuclear-net.hpp"

namespace cuda_util {
	/// function to move a buffer to the GPU
	/**
	 * TODO
	 */
	template<class T>
	T *move_to_gpu(const T* const ptr, int dimension) {
		T *dev_ptr;
		
		cudaMalloc((void**)&dev_ptr, dimension*sizeof(T));
		cudaMemcpy(dev_ptr, ptr, dimension*sizeof(T), cudaMemcpyHostToDevice);

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
			cudaFree((void*)ptr_reaction_list::reactant_product);
			cudaFree((void*)ptr_reaction_list::reactant_begin);
			cudaFree((void*)ptr_reaction_list::product_begin);
		}
	};

	
	/// function to move reactions to the GPU
	/**
	 * TODO
	 */
	gpu_reaction_list move_to_gpu(const ptr_reaction_list &reactions) {
		gpu_reaction_list dev_reactions;
		dev_reactions.num_reactions = reactions.num_reactions;

		dev_reactions.reactant_product = cuda_util::move_to_gpu(reactions.reactant_product, reactions.reactant_begin[reactions.num_reactions]);
		dev_reactions.reactant_begin   = cuda_util::move_to_gpu(reactions.reactant_begin,   reactions.num_reactions + 1);
		dev_reactions.product_begin    = cuda_util::move_to_gpu(reactions.product_begin,    reactions.num_reactions);

		return dev_reactions;
	} 
}