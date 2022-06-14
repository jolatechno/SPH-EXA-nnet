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
		
		cudaMalloc(&dev_ptr, dimension*sizeof(T));
		cudaMemcpy(dev_ptr, ptr, dimension*sizeof(T), cudaMemcpyHostToDevice);

		return dev_ptr;
	}
}


namespace nnet {
	/// function to move reactions to the GPU
	/**
	 * TODO
	 */
	std::shared_ptr<ptr_reaction_list> move_to_gpu(const ptr_reaction_list &reactions) {
		ptr_reaction_list dev_reactions;
		dev_reactions.num_reactions = reactions.num_reactions;

		dev_reactions.reactant_product = cuda_util::move_to_gpu(reactions.reactant_product, reactions.reactant_begin[reactions.num_reactions]);
		dev_reactions.reactant_begin   = cuda_util::move_to_gpu(reactions.reactant_begin,   reactions.num_reactions);
		dev_reactions.product_begin    = cuda_util::move_to_gpu(reactions.product_begin,    reactions.num_reactions + 1);

		std::shared_ptr<ptr_reaction_list> ptr = std::make_shared<ptr_reaction_list>(dev_reactions);

		auto del_ptr = std::get_deleter<void(*)(ptr_reaction_list*)>(ptr);
		*del_ptr = [](ptr_reaction_list *ptr_) {
			cudaFree((void*)ptr_->reactant_product);
			cudaFree((void*)ptr_->reactant_begin);
			cudaFree((void*)ptr_->product_begin);
		};

		return ptr;
	} 
}