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
 * @brief Definition of CUDA integration functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#include "nuclear-net.cuh"

namespace sphexa {
namespace sphnnet {
	/***********************************************/
	/* code to compute nuclear reaction on the GPU */
	/***********************************************/


	/*! @brief kernel that integrate nuclear reaction over a given timestep in parallel on device
	 * 
	 * called in cudaComputeNuclearReactions, should not be directly accessed by user
	 */
	template<class func_type, class func_eos, typename Float>
	__global__ void cudaKernelComputeNuclearReactions(const size_t n_particles, const int dimension,
		Float *global_buffer,
		Float *rho_, Float *previous_rho_, Float **Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list *reactions, const func_type *construct_rates_BE, const func_eos *eos,
		bool use_drhodt)
	{
	    size_t block_begin =                         blockIdx.x*blockDim.x*constants::cuda_num_iteration_per_thread;
	    size_t block_end   = (blockIdx.x + 1)*blockDim.x*constants::cuda_num_iteration_per_thread;
	    if (block_end > n_particles)
	    	block_end = n_particles;
	    size_t block_size  = block_end - block_begin;

	    // task status
	    static const int task_free_status     = -3;
	    static const int task_loading_status  = -2;
	    static const int task_finished_status = -1;
	    static const int task_running_status  =  0;
	    // thread status
	    static const int thread_seaking_status = -1;
	    static const int thread_running_status =  0;

	    // allocating shared array
	    __shared__ int task_status[constants::cuda_num_thread_per_block_nnet*constants::cuda_num_iteration_per_thread];
	    __shared__ int thread_status[constants::cuda_num_thread_per_block_nnet];
	    __shared__ bool exit;
	    // assigning shared array
	    for (int i =       threadIdx.x*constants::cuda_num_iteration_per_thread;
	    	     i < (threadIdx.x + 1)*constants::cuda_num_iteration_per_thread;
	    	     ++i)
	    {
	    	task_status[i] = task_free_status;
	    }
	    thread_status[threadIdx.x] = thread_seaking_status;

	    // buffer sizes
	    const size_t Y_size        =  dimension;
	    const size_t Mp_size       = (dimension + 1)*(dimension + 1);
	    const size_t RHS_size      =  dimension + 1;
	    const size_t DY_T_size     =  dimension + 1;
	    const size_t Y_buffer_size =  dimension;
	    const size_t rates_size    =  reactions->size();

	    // allocate local buffer
	    Float T_buffer;
	    Float *Y         = global_buffer + (Y_size + Mp_size + RHS_size + DY_T_size + Y_buffer_size + rates_size)*(blockIdx.x*blockDim.x + threadIdx.x);
    	Float *Mp        = Y        + Y_size;
		Float *RHS       = Mp       + Mp_size;
		Float *DY_T      = RHS      + RHS_size;
		Float *Y_buffer  = DY_T     + DY_T_size;
		Float *rates     = Y_buffer + Y_buffer_size;

		// initial condition
		int iter = 1;
		Float elapsed = 0.0;
		// run simulation
		while (true) {
			/* !!!!!!!!!!!!!!!!!!!!!!!!
			       work sharing
			!!!!!!!!!!!!!!!!!!!!!!!! */

			__syncthreads();

			if (threadIdx.x == 0) {
				exit = true;
				int task_idx = 0;
				for (int thread_idx = 0; thread_idx < constants::cuda_num_thread_per_block_nnet; ++thread_idx)
					if (thread_status[thread_idx] == thread_seaking_status) {
						// find next free task
						for (; task_idx < block_size; ++task_idx)
							if (task_status[task_idx] == task_free_status) {
								// assign task
								thread_status[thread_idx] = task_idx;
								task_status[task_idx]     = task_loading_status;

								// increase task_idx
								++task_idx;
								// can't exit as a task is still running
								exit = false;

								// exit loop
								break;
							}
					} else {
						// can't exit as a task is still running
						exit = false;
					}
			}

			__syncthreads();

			if (exit)
				break;

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			     actual simulation
			!!!!!!!!!!!!!!!!!!!!!!!! */


			for (int i = 0; i < constants::cuda_iteration_between_work_resharing; ++i)
				if (thread_status[threadIdx.x] >= thread_running_status) {
					const int shared_idx = thread_status[threadIdx.x];
					const size_t idx = block_begin + shared_idx;

					// copy Y to local buffer
					if (task_status[shared_idx] == task_loading_status) {
						for (int j = 0; j < dimension; ++j)
							Y[j] = Y_[j][idx];
						task_status[shared_idx] = task_running_status;
					}

					// compute drho/dt
					Float drho_dt = 0;
					if (use_drhodt)
						previous_rho_[idx] = (rho_[idx] - previous_rho_[idx])/previous_dt;

					// generate system
					nnet::prepare_system_substep(dimension,
						Mp, RHS, rates,
						*reactions, *construct_rates_BE, *eos,
						Y, temp_[idx], Y_buffer, T_buffer,
						rho_[idx], drho_dt,
						hydro_dt, elapsed, dt_[idx], iter);

					// solve M*D{T, Y} = RHS
					eigen::solve(Mp, RHS, DY_T, dimension + 1, (Float)nnet::constants::epsilon_system);

					// finalize
					if(nnet::finalize_system_substep(dimension,
						Y, temp_[idx],
						Y_buffer, T_buffer,
						DY_T, hydro_dt, elapsed,
						dt_[idx], iter))
					{
						// copy Y "buffer" back to actual storage
						for (int j = 0; j < dimension; ++j)
							Y_[j][idx] = Y[j];

						// reset
						task_status[shared_idx]    = task_finished_status;
						thread_status[threadIdx.x] = thread_seaking_status;
						iter = 0;
						elapsed = 0.0;
					}

					++iter;
				}
		}
	}


	/*! @brief function that integrate nuclear reaction over a given timestep in parallel on device
	 * 
	 * used in include/nnet/sphexa/nuclear-net.hpp, should not be directly accessed by user
	 */
	template<class func_type, class func_eos, typename Float>
	void cudaComputeNuclearReactions(const size_t n_particles, const int dimension,
		thrust::device_vector<Float> &buffer,
		Float *rho_, Float *previous_rho_, Float **Y_, Float *temp_, Float *dt_,
		const Float hydro_dt, const Float previous_dt,
		const nnet::gpu_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		bool use_drhodt)
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
		const size_t Y_size        =  dimension;
	    const size_t Mp_size       = (dimension + 1)*(dimension + 1);
	    const size_t RHS_size      =  dimension + 1;
	    const size_t DY_T_size     =  dimension + 1;
	    const size_t Y_buffer_size =  dimension;
	    const size_t rates_size    =  reactions.size();
		// allocate global buffer
	    const size_t buffer_size = (Y_size + Mp_size + RHS_size + DY_T_size + Y_buffer_size + rates_size)*cuda_num_blocks*constants::cuda_num_thread_per_block_nnet;
		if (buffer.size() < buffer_size)
			buffer.resize(buffer_size);


		// launch kernel
	    cudaKernelComputeNuclearReactions<<<cuda_num_blocks, constants::cuda_num_thread_per_block_nnet>>>(n_particles, dimension,
	(Float*)thrust::raw_pointer_cast(buffer.data()),
			rho_, previous_rho_, Y_, temp_, dt_,
			hydro_dt, previous_dt,
			dev_reactions, dev_construct_rates_BE, dev_eos,
			use_drhodt);


		// debuging: check for error
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

	    // free cuda classes
	    gpuErrchk(cudaFree(dev_reactions));
	    gpuErrchk(cudaFree(dev_construct_rates_BE));
	    gpuErrchk(cudaFree(dev_eos));
	}


	// used templates:
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net87::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net86::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net14::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);

	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net87::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<double> const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net86::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<double> const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<double>&, double*, double*, double**, double*, double*, const double, const double,
		nnet::gpu_reaction_list const&, nnet::net14::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<double> const&,
		bool);

	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net87::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net86::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net14::compute_reaction_rates_functor const&, nnet::eos::ideal_gas_functor const&,
		bool);

	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net87::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<float> const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net86::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<float> const&,
		bool);
	template void cudaComputeNuclearReactions(const unsigned long, const int,
		thrust::device_vector<float>&, float*, float*, float**, float*, float*, const float, const float,
		nnet::gpu_reaction_list const&, nnet::net14::compute_reaction_rates_functor const&, nnet::eos::helmholtz_functor<float> const&,
		bool);



	/************************************************************/
	/* code to compute helmholtz equation of a state on the GPU */
	/************************************************************/


	/*! @brief kernel that computes helmholtz EOS in parallel on device
	 * 
	 * called in cudaComputeHelmholtz, should not be directly accessed by user
	 */
	template<typename Float /*, class func_eos*/>
	__global__ void cudaKernelComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, Float *const* Y_, 
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT)
	{
		size_t thread = blockIdx.x*blockDim.x + threadIdx.x;
		if (thread < n_particles) {
			// compute abar and zbar
			double abar = 0, zbar = 0;
			for (int i = 0; i < dimension; ++i) {
				abar += Y_[i][thread];
				zbar += Y_[i][thread]*Z[i];
			}


			// actually compute helmholtz eos
			auto eos_struct = nnet::eos::helmholtz(abar, zbar, temp_[thread], rho_[thread]);


			// copy results to buffers
			u[thread]    = eos_struct.u;
			cv[thread]   = eos_struct.cv;
			p[thread]    = eos_struct.p;
			c[thread]    = eos_struct.c;
			dpdT[thread] = eos_struct.dpdT;
		}
	}


	/*! @brief function that computes helmholtz EOS in parallel on device
	 * 
	 * used in include/nnet/sphexa/nuclear-net.hpp, should not be directly accessed by user
	 */
	template<typename Float>
	void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const Float *Z,
		const Float *temp_, const Float *rho_, Float *const* Y_,
		Float *u, Float *cv, Float *p, Float *c, Float *dpdT)
	{
		// compute chunk sizes
		int cuda_num_blocks = (n_particles + constants::cuda_num_thread_per_block - 1)/constants::cuda_num_thread_per_block;
		
		// launch kernel
	    cudaKernelComputeHelmholtz<<<cuda_num_blocks, constants::cuda_num_thread_per_block>>>(n_particles, dimension, Z,
			temp_, rho_, Y_,
			u, cv, p, c, dpdT);
	}


	// used templates:
	template void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const double *Z,
		const double *temp_, const double *rho_, double *const* Y_,
		double *u, double *cv, double *p, double *c, double *dpdT);
	template void cudaComputeHelmholtz(const size_t n_particles, const int dimension, const float *Z,
		const float *temp_, const float *rho_, float *const* Y_,
		float *u, float *cv, float *p, float *c, float *dpdT);
	}
}