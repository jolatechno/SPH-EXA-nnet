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
 * @brief Interface definition for CUDA integration functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "sph/data_util.hpp"
#include "sph/field_states.hpp"
#include "sph/traits.hpp"

#include "cstone/util/util.hpp"
#include "cstone/util/array.hpp"

#include "../../nuclear-net.hpp"
#include "../../CUDA/nuclear-net.hpp"

#include "../../eos/helmholtz.hpp"
#include "../util/algorithm.hpp"

#include "../../net14/net14.hpp"
#include "../../net86/net86.hpp"
#include "../../net87/net87.hpp"
#include "../../eos/ideal_gas.hpp"
#include "../../eos/helmholtz.hpp"

#ifndef CUDA_NUM_ITERATION_PER_THREAD
	#define CUDA_NUM_ITERATION_PER_THREAD 8
#endif
#ifndef CUDA_NUM_THREAD_PER_BLOCK
	#define CUDA_NUM_THREAD_PER_BLOCK 32
#endif
#ifndef CUDA_NUM_THREAD_PER_BLOCK_NNET
	#define CUDA_NUM_THREAD_PER_BLOCK_NNET CUDA_NUM_THREAD_PER_BLOCK
#endif

namespace sphexa::sphnnet {
	namespace constants {
		/// number of consecutive iteration per cuda thread
		const int cuda_num_iteration_per_thread = CUDA_NUM_ITERATION_PER_THREAD;
		/// number of thread per cuda thread block for nuclear network
		const int cuda_num_thread_per_block_nnet = CUDA_NUM_THREAD_PER_BLOCK_NNET;
		/// number of thread per cuda thread block
		const int cuda_num_thread_per_block = CUDA_NUM_THREAD_PER_BLOCK;
	}

	template<class func_type, class func_eos, typename Float>
	extern void cudaComputeNuclearReactions(const size_t n_particles, const int dimension,
		thrust::device_vector<Float> &buffer,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		bool use_drhodt);

	template<typename Float>
	extern void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_,
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT);
}