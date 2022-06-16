#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// #include "nuclear-net.cuh"
#include "../../nuclear-net.hpp"


namespace sphexa {
namespace sphnnet {
	namespace constants {
		/// number of consecutive iteration per cuda thread
		const int cuda_num_iteration_per_thread = 16;
		/// number of thread per cuda thread block
		const int cuda_num_thread_per_block = 128;
	}

	template<class func_type, class func_eos, typename Float>
	__global__ void cudaKernelComputeNuclearReactions(const int n_particles, const int dimension,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list *reactions, const func_type *construct_rates_BE, const func_eos *eos)
	{
	    size_t thread = blockIdx.x*blockDim.x + threadIdx.x;
	    size_t begin  =       thread*constants::cuda_num_iteration_per_thread;
	    size_t end    = (thread + 1)*constants::cuda_num_iteration_per_thread;
	    if (end > n_particles)
	    	end = n_particles; 

	    Y_[0] = (Float)reactions->size();

    	Float *Mp        = new Float[(dimension + 1)*(dimension + 1)];
		Float *RHS       = new Float[                (dimension + 1)];
		Float *DY_T      = new Float[                (dimension + 1)];
		Float *Y_buffer  = new Float[                      dimension];
		/*Float *rates     = new Float[reactions->size()];
		Float *drates_dT = new Float[reactions->size()];*/

		/*for (size_t i = begin; i < end; ++i)
		    if (rho_[i] > nnet::constants::min_rho && temp_[i] > nnet::constants::min_temp) {
				// compute drho/dt
				Float drho_dt = previous_rho_[i] <= 0 ? 0. : (rho_[i] - previous_rho_[i])/previous_dt;

				// solve
				nnet::solve_system_substep(dimension,
					Mp, RHS, DY_T, rates, drates_dT,
					*reactions, *construct_rates_BE, *eos,
					Y_ + dimension*i, temp_[i], Y_buffer,
					rho_[i], drho_dt, hydro_dt, dt_[i]);
			}*/




		// only test "eos"
		auto eos_struct = (*eos)(Y_buffer, (Float)1.e9, (Float)1e9);
		temp_[1] = eos_struct.cv;



		// only test "construct_rates_BE"
		/*(*construct_rates_BE)(Y_, (Float)1.e9, (Float)1e9, eos_struct, &Mp[1], rates, drates_dT);
		temp_[2] = rates[0];
		temp_[3] = drates_dT[0];*/





		delete[] Mp;
		delete[] RHS;
		delete[] DY_T;
		delete[] Y_buffer;
		/*delete[] rates;
		delete[] drates_dT;*/
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const int n_particles, const int dimension,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
	{
		// copy classes to gpu
		nnet::gpu_reaction_list *dev_reactions;
		func_type               *dev_construct_rates_BE;
		func_eos                *dev_eos;
		// allocate
		gpuErrchk(cudaMalloc((void**)&dev_reactions,          sizeof(nnet::gpu_reaction_list)));
		gpuErrchk(cudaMalloc((void**)&dev_construct_rates_BE, sizeof(func_type)));
		gpuErrchk(cudaMalloc((void**)&dev_eos,                sizeof(func_eos)));
		// actually copy
		gpuErrchk(cudaMemcpy(dev_reactions,          &reactions,          sizeof(nnet::gpu_reaction_list), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_construct_rates_BE, &construct_rates_BE, sizeof(func_type),               cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_eos,                &eos,                sizeof(func_eos),                cudaMemcpyHostToDevice));

		// compute chunk sizes
		int num_threads     = (n_particles + constants::cuda_num_iteration_per_thread - 1)/constants::cuda_num_iteration_per_thread;
		int cuda_num_blocks = (num_threads + constants::cuda_num_thread_per_block     - 1)/constants::cuda_num_thread_per_block;

		// launch kernel
	    cudaKernelComputeNuclearReactions<<<cuda_num_blocks, constants::cuda_num_thread_per_block>>>(n_particles, dimension,
			rho_, previous_rho_, Y_, temp_, dt_,
			hydro_dt, previous_dt,
			dev_reactions, dev_construct_rates_BE, dev_eos);
	}
}
}