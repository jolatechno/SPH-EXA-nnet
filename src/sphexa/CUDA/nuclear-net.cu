#pragma once

#include "nuclear-net.cuh"

namespace sphexa {
namespace sphnnet {
	/***********************************************/
	/* code to compute nuclear reaction on the GPU */
	/***********************************************/



	template<class func_type, class func_eos, typename Float>
	__global__ void cudaKernelComputeNuclearReactions(const size_t n_particles, const int dimension,
		Float *global_buffer,
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

	    // buffer sizes
	    const int Mp_size       = (dimension + 1)*(dimension + 1);
	    const int RHS_size      =  dimension + 1;
	    const int DY_T_size     =  dimension + 1;
	    const int Y_buffer_size =  dimension;
	    const int rates_size    =  reactions->size();

	    // allocate local buffer
	    Float T_buffer;
	    Float *Buffer    = global_buffer + (Mp_size + RHS_size + DY_T_size + Y_buffer_size + rates_size)*(blockIdx.x*blockDim.x + threadIdx.x);
    	Float *Mp        = Buffer;
		Float *RHS       = Buffer + Mp_size;
		Float *DY_T      = Buffer + Mp_size + RHS_size;
		Float *Y_buffer  = Buffer + Mp_size + RHS_size + DY_T_size;
		Float *rates     = Buffer + Mp_size + RHS_size + DY_T_size + Y_buffer_size;

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
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const size_t n_particles, const int dimension,
		thrust::device_vector<Float> &buffer,
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
		int num_threads     = (n_particles + constants::cuda_num_iteration_per_thread  - 1)/constants::cuda_num_iteration_per_thread;
		int cuda_num_blocks = (num_threads + constants::cuda_num_thread_per_block_nnet - 1)/constants::cuda_num_thread_per_block_nnet;

		// buffer sizes
	    const int Mp_size       = (dimension + 1)*(dimension + 1);
	    const int RHS_size      =  dimension + 1;
	    const int DY_T_size     =  dimension + 1;
	    const int Y_buffer_size =  dimension;
	    const int rates_size    =  reactions.size();
		// allocate global buffer
	    const size_t buffer_size = (Mp_size + RHS_size + DY_T_size + Y_buffer_size + rates_size)*cuda_num_blocks*constants::cuda_num_thread_per_block_nnet;
		if (buffer.size() < buffer_size)
			buffer.resize(buffer_size);

		// launch kernel
	    cudaKernelComputeNuclearReactions<<<cuda_num_blocks, constants::cuda_num_thread_per_block_nnet>>>(n_particles, dimension,
	(Float*)thrust::raw_pointer_cast(buffer.data()),
			rho_, previous_rho_, Y_, temp_, dt_,
			hydro_dt, previous_dt,
			dev_reactions, dev_construct_rates_BE, dev_eos);

	    // free cuda classes
	    gpuErrchk(cudaFree(dev_reactions));
	    gpuErrchk(cudaFree(dev_construct_rates_BE));
	    gpuErrchk(cudaFree(dev_eos));
	}



	/************************************************************/
	/* code to compute helmholtz equation of a state on the GPU */
	/************************************************************/



	template<typename Float /*, class func_eos*/>
	__global__ void cudaKernelComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_, 
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT)
	{
		size_t thread = blockIdx.x*blockDim.x + threadIdx.x;
		if (thread < n_particles) {
			// compute abar and zbar
			double abar = algorithm::accumulate(Y_ + thread*dimension, Y_ + (thread + 1)*dimension, (double)0.);
			double zbar = eigen::dot(Y_ + thread*dimension, Y_ + (thread + 1)*dimension, Z);

			auto eos_struct = nnet::eos::helmholtz(abar, zbar, temp_[thread], rho_[thread]);

			u[thread]    = eos_struct.u;
			cv[thread]   = eos_struct.cv;
			p[thread]    = eos_struct.p;
			c[thread]    = eos_struct.c;
			dpdT[thread] = eos_struct.dpdT;
		}
	}


	template<typename Float>
	__host__ void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, const Float *Y_,
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT)
	{
		// compute chunk sizes
		int cuda_num_blocks = (n_particles + constants::cuda_num_thread_per_block - 1)/constants::cuda_num_thread_per_block;
		
		// launch kernel
	    cudaKernelComputeHelmholtz<<<cuda_num_blocks, constants::cuda_num_thread_per_block>>>(n_particles, dimension, Z,
			temp_, rho_, Y_,
			u, cv, p, c, dpdT);
	}
}
}