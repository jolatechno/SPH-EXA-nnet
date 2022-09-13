/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief CUDA utility functions.
 */


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