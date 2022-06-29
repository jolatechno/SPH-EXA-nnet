#pragma once

#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// #include "nuclear-net.cuh"
#include "../../nuclear-net.hpp"
#include "../util/algorithm.hpp"


namespace sphexa {
namespace sphnnet {
	namespace constants {
		/// number of consecutive iteration per cuda thread
		const int cuda_num_iteration_per_thread = 8;
		/// number of thread per cuda thread block
		const int cuda_num_thread_per_block = 32;
	}



	/***********************************************/
	/* code to compute nuclear reaction on the GPU */
	/***********************************************/



	template<class func_type, class func_eos, typename Float>
	__global__ void cudaKernelComputeNuclearReactions(const size_t n_particles, const int dimension,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list *reactions, const func_type *construct_rates_BE, const func_eos *eos)
	{
	    const size_t block_begin =                               blockIdx.x*blockDim.x*constants::cuda_num_iteration_per_thread;
	    const size_t block_end   = algorithm::min((size_t)((blockIdx.x + 1)*blockDim.x*constants::cuda_num_iteration_per_thread), n_particles);
	    const size_t block_size  = block_end - block_begin;

	    // satatus
	    static const int free_status     = -2;
	    static const int finished_status = -1;
	    static const int running_status  =  0;

	    // initialized shared array
	    __shared__ int status[constants::cuda_num_thread_per_block*constants::cuda_num_iteration_per_thread];
	     for (int i =                         constants::cuda_num_iteration_per_thread*threadIdx.x;
	    		 i < algorithm::min((size_t)(constants::cuda_num_iteration_per_thread*(threadIdx.x + 1)), block_size);
	    	   ++i)
	    {
	     	status[i] = free_status;
	    }

	    // allocate local buffer
	    Float T_buffer;
    	Float *Mp        = new Float[(dimension + 1)*(dimension + 1)];
		Float *RHS       = new Float[                (dimension + 1)];
		Float *DY_T      = new Float[                (dimension + 1)];
		Float *Y_buffer  = new Float[                      dimension];
		Float *rates     = new Float[reactions->size()];

		// run simulation 
		int iter = 1;
		int shared_idx = -1;
		Float elapsed = 0.0;
		bool did_not_find = false;
		while (true) {
			/* !!!!!!!!!!!!!!!!!!!!!!!!
			       work sharing
			!!!!!!!!!!!!!!!!!!!!!!!! */
			__syncthreads();
			bool exit = shared_idx == -1;
			if (did_not_find) {
				for (int i = 0; i < block_size; ++i)
					if (status[i] != finished_status) {
						exit = false;
						break;
					}
			} else if (shared_idx == -1) {
				int num_free = 0, thread_id = threadIdx.x;
				for (int i = 0; i < block_size; ++i)
					if (status[i] == free_status) {
						exit = false;
						++num_free;

						// acquire n-th free ttask
						if (num_free == thread_id + 1) {
							shared_idx = i;
							break;
						}
					} else if (status[i] >= running_status) {
						exit = false;

						if (status[i] < threadIdx.x)
							--thread_id;
					}
				if (shared_idx == -1)
					did_not_find = true;
			}
			// exit condition
			if (exit)
				break;
			__syncthreads();

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			     actual simulation
			!!!!!!!!!!!!!!!!!!!!!!!! */
			if (shared_idx >= 0) {
				status[shared_idx] = threadIdx.x;
				const size_t idx = block_begin + shared_idx;

				// compute drho/dt
				Float drho_dt = previous_rho_[idx] <= 0 ? 0. : (rho_[idx] - previous_rho_[idx])/previous_dt;

				// generate system
				nnet::prepare_system_substep(dimension,
					Mp, RHS, rates,
					*reactions, *construct_rates_BE, *eos,
					Y_ + dimension*idx, temp_[idx], Y_buffer, T_buffer,
					rho_[idx], drho_dt,
					hydro_dt, elapsed, dt_[idx], iter);

				// solve M*D{T, Y} = RHS
				eigen::solve(Mp, RHS, DY_T, dimension + 1, nnet::constants::epsilon_system);

				// finalize
				if(nnet::finalize_system_substep(dimension,
					Y_ + dimension*idx, temp_[idx],
					Y_buffer, T_buffer,
					DY_T, hydro_dt, elapsed,
					dt_[idx], iter))
				{
					// reset
					status[shared_idx] = finished_status;
					iter = 0;
					shared_idx = -1;
					elapsed = 0.0;
				}

				++iter;
			}
		}

		// free buffers
		delete[] Mp;
		delete[] RHS;
		delete[] DY_T;
		delete[] Y_buffer;
		delete[] rates;
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const size_t n_particles, const int dimension,
		Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
	{
		// insure that the heap is large enough
		size_t cuda_heap_limit, requiered_heap = (dimension*dimension*3 + dimension)*n_particles;
		gpuErrchk(cudaDeviceGetLimit(&cuda_heap_limit, cudaLimitMallocHeapSize));
		if (cuda_heap_limit < requiered_heap)
			gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, requiered_heap));

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



	/************************************************************/
	/* code to compute helmholtz equation of a state on the GPU */
	/************************************************************/



	template<typename Float /*, class func_eos*/>
	__global__ void cudaKernelComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_, 
		Float *cv, Float *p, Float *c /*,
		const func_eos *eos*/)
	{
		size_t thread = blockIdx.x*blockDim.x + threadIdx.x;
		if (thread < n_particles) {
			// compute abar and zbar
			double abar = algorithm::accumulate(Y_ + thread*dimension, Y_ + (thread + 1)*dimension, (double)0.);
			double zbar = eigen::dot(Y_ + thread*dimension, Y_ + (thread + 1)*dimension, Z);

			auto eos_struct = /*(*eos)*/ nnet::eos::helmholtz(abar, zbar, temp_[thread], rho_[thread]);

		 // u[thread]  = eos_struct.u;
			cv[thread] = eos_struct.cv;
			p[thread]  = eos_struct.p;
			c[thread]  = eos_struct.c;
		}
	}


	template<typename Float>
	__host__ void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_,
		Float *cv, Float *p, Float *c)
	{
		/* // copy classes to gpu
		nnet::eos::helmholtz_function *dev_eos;
		// allocate
		gpuErrchk(cudaMalloc((void**)&dev_eos, sizeof(nnet::eos::helmholtz_function)));
		// actually copy
		gpuErrchk(cudaMemcpy(dev_eos, &nnet::eos::helmholtz, sizeof(nnet::eos::helmholtz_function), cudaMemcpyHostToDevice)); */
		
		// compute chunk sizes
		int cuda_num_blocks = (n_particles + constants::cuda_num_thread_per_block - 1)/constants::cuda_num_thread_per_block;
		
		// launch kernel
	    cudaKernelComputeHelmholtz<<<cuda_num_blocks, constants::cuda_num_thread_per_block>>>(n_particles, dimension, Z,
			temp_, rho_, Y_,
			cv, p, c/*,
			dev_eos*/);
	}
}
}