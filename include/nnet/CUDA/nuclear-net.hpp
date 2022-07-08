#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "../nuclear-net.hpp"
#include "cuda-util.hpp"

namespace nnet {
	/// class for reactions on gpu
	/**
	 * TODO
	 */
	class gpu_reaction_list : public ptr_reaction_list {
	private:
		friend gpu_reaction_list move_to_gpu(const ptr_reaction_list &reactions);
		friend void inline free(gpu_reaction_list &reactions);

	public:
		gpu_reaction_list() {}
		~gpu_reaction_list() {};
	};


	/// function to move reactions to the GPU
	/**
	 * TODO
	 */
	gpu_reaction_list inline move_to_gpu(const ptr_reaction_list &reactions) {
		gpu_reaction_list dev_reactions;
		dev_reactions.num_reactions = reactions.num_reactions;

		dev_reactions.reactant_product = cuda_util::move_to_gpu<reaction::reactant_product>(reactions.reactant_product, reactions.reactant_begin[reactions.num_reactions]);
		dev_reactions.reactant_begin   = cuda_util::move_to_gpu<int>(reactions.reactant_begin, reactions.num_reactions + 1);
		dev_reactions.product_begin    = cuda_util::move_to_gpu<int>(reactions.product_begin,  reactions.num_reactions);

		return dev_reactions;
	} 


	/// function to free reactions
	/**
	 * TODO
	 */
	void inline free(gpu_reaction_list &reactions) {
		gpuErrchk(cudaFree((void*)reactions.reactant_product));
		gpuErrchk(cudaFree((void*)reactions.reactant_begin));
		gpuErrchk(cudaFree((void*)reactions.product_begin));
	}
}