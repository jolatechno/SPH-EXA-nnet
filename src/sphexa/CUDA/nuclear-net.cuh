#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "../../nuclear-net.hpp"
#include "../../CUDA/nuclear-net.hpp"
#include "../../eos/helmholtz.hpp"
#include "../util/algorithm.hpp"

#ifndef CUDA_NUM_ITERATION_PER_THREAD
	#define CUDA_NUM_ITERATION_PER_THREAD 8
#endif
#ifndef CUDA_NUM_THREAD_PER_BLOCK_NNET
	#define CUDA_NUM_THREAD_PER_BLOCK_NNET 32
#endif
#ifndef CUDA_NUM_THREAD_PER_BLOCK
	#define CUDA_NUM_THREAD_PER_BLOCK 32
#endif

namespace sphexa {
namespace sphnnet {
	namespace constants {
		/// number of consecutive iteration per cuda thread
		const int cuda_num_iteration_per_thread = CUDA_NUM_ITERATION_PER_THREAD;
		/// number of thread per cuda thread block for nuclear network
		const int cuda_num_thread_per_block_nnet = CUDA_NUM_THREAD_PER_BLOCK_NNET;
		/// number of thread per cuda thread block
		const int cuda_num_thread_per_block = CUDA_NUM_THREAD_PER_BLOCK;
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const size_t n_particles, const int dimension,
		thrust::device_vector<Float> &buffer,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos);

	template<typename Float>
	__host__ void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_,
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT);
}
}