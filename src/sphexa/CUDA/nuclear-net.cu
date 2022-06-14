#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "nuclear-net.cuh"

#ifndef CUDA_BLOCK_SIZE
	#define CUDA_BLOCK_SIZE 256
#endif

namespace sphexa {
namespace sphnnet {
	template<class func_type, class func_eos, typename Float>
	__global__ void cudaKernelComputeNuclearReactions(const int n_particles, const int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
	{
	    const int i = blockIdx.x*blockDim.x + threadIdx.x;
	    if (i < n_particles) {
	    	Float *Mp        = (Float*)malloc((dimension + 1)*(dimension + 1)*sizeof(Float));
			Float *RHS       = (Float*)malloc(                (dimension + 1)*sizeof(Float));
			Float *DY_T      = (Float*)malloc(                (dimension + 1)*sizeof(Float));
			Float *Y_buffer  = (Float*)malloc(                      dimension*sizeof(Float));
			Float *rates     = (Float*)malloc(reactions.size()*sizeof(Float));
			Float *drates_dT = (Float*)malloc(reactions.size()*sizeof(Float));

		    if (rho_[i] > nnet::constants::min_rho && temp_[i] > nnet::constants::min_temp) {
				// compute drho/dt
				Float drho_dt = previous_rho_[i] <= 0 ? 0. : (rho_[i] - previous_rho_[i])/previous_dt;

				// solve
				nnet::solve_system_substep(dimension,
					Mp, RHS, DY_T, rates, drates_dT,
					reactions, construct_rates_BE, eos,
					&Y_[dimension*i], temp_[i], Y_buffer,
					rho_[i], drho_dt, hydro_dt, dt_[i]);
			}

			free(Mp);
			free(RHS);
			free(DY_T);
			free(Y_buffer);
			free(rates);
			free(drates_dT);
		}
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const int n_particles, const int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
	{
		int cuda_n_thread_per_block = CUDA_BLOCK_SIZE;
		int cuda_n_blocks           = (n_particles + cuda_n_thread_per_block - 1) / cuda_n_thread_per_block;

	    cudaKernelComputeNuclearReactions<<<cuda_n_blocks, cuda_n_thread_per_block>>>(n_particles, dimension,
			rho_, previous_rho_, Y_, temp_, dt_,
			hydro_dt, previous_dt,
			reactions, construct_rates_BE, eos);
	}
}
}