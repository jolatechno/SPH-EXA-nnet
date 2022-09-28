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
 * @brief Observables for nuclear networks (mainly energy).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include "../CUDA/cuda.inl"
#if COMPILE_DEVICE
	#include <device_launch_parameters.h>
	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include <cuda_runtime.h>

	#include <thrust/device_vector.h>

	#include "../CUDA/nuclear-net.hpp"
	#include "CUDA/nuclear-net.cuh"
#endif

#include <numeric>
#include <omp.h>

#include "nuclear-data.hpp"
#include "../eos/helmholtz.hpp"

#include "../eigen/eigen.hpp"
#include "util/algorithm.hpp"

#include "mpi/mpi-wrapper.hpp"

#include "sph/data_util.hpp"

#include "../nuclear-net.hpp"

namespace sphexa::sphnnet {
	/*! @brief function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y
	 * 
	 * @param n                   nuclearDataType including a field of nuclear abundances "Y"
     * @param firstIndex          first (included) particle considered in n
	 * @param lastIndex           last (excluded) particle considered in n
     * @param hydro_dt            integration timestep
     * @param previous_dt         previous integration timestep
     * @param reactions           reaction list
     * @param construct_rates_BE  function constructing rates, rate derivatives and binding energies
     * @param eos                 equation of state
     * @param use_drhodt          if true considers drho/dt in eos
     * @param jumpToNse           function to jump to nuclear statistical equilibrium
     */
	template<class Data, class func_type, class func_eos, typename Float, class nseFunction=void*>
	void computeNuclearReactions(Data &n, size_t firstIndex, size_t lastIndex, const Float hydro_dt, const Float previous_dt,
		const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		bool use_drhodt,
		const nseFunction jumpToNse=NULL)
	{
		n.minDt_m1 = n.minDt;
		n.minDt    = hydro_dt;
		n.ttot    += n.minDt;
		++n.iteration;

		const size_t n_particles = n.temp.size();
		const int dimension = n.Y.size();
		
		if constexpr (HaveGpu<typename Data::AcceleratorType>{} && COMPILE_DEVICE) {

			/* !!!!!!!!!!!!!
			GPU non-batch solver
			!!!!!!!!!!!!! */

#if COMPILE_DEVICE
			// check for drho/dt allocation
			Float* previous_rho_ptr = nullptr;
			if (use_drhodt) {
				int previous_rho_idx = std::distance(n.devData.fieldNames.begin(),
					std::find(n.devData.fieldNames.begin(), n.devData.fieldNames.end(), "previous_rho"));
				if (!n.devData.isAllocated(previous_rho_idx)) {
					use_drhodt = false;
					std::cerr << "disabeling using drho/dt because 'previous_rho' isn't alocated !\n";
				}
			}
			if (use_drhodt)
				previous_rho_ptr = (Float*)thrust::raw_pointer_cast(n.devData.previous_rho.data() + firstIndex);

			// reactions to GPU
			nnet::gpu_reaction_list dev_reactions = nnet::move_to_gpu(reactions);

			// copy pointers to GPU
			std::vector<Float*> Y_raw_ptr(dimension);
			Float **Y_dev_ptr;
			gpuErrchk(cudaMalloc((void**)&Y_dev_ptr, dimension*sizeof(Float*)));
			for (int i = 0; i < dimension; ++i)
	            Y_raw_ptr[i] = (Float*)thrust::raw_pointer_cast(n.devData.Y[i].data() + firstIndex); // store Y raw pointer to CPU
	        gpuErrchk(cudaMemcpy((void*)Y_dev_ptr, (void*)Y_raw_ptr.data(), dimension*sizeof(Float*), cudaMemcpyHostToDevice));


			// call the cuda kernel wrapper
			cudaComputeNuclearReactions(lastIndex - firstIndex, dimension,
				n.devData.buffer,
		(Float*)thrust::raw_pointer_cast(n.devData.rho.data()  + firstIndex),
		        previous_rho_ptr,
		        Y_dev_ptr,
		(Float*)thrust::raw_pointer_cast(n.devData.temp.data() + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.dt.data()   + firstIndex),
				hydro_dt, previous_dt,
				dev_reactions, construct_rates_BE, eos,
				use_drhodt);


			// free cuda buffer
			gpuErrchk(cudaFree((void*)Y_dev_ptr));
			free(dev_reactions);
#endif
		} else {

			/* !!!!!!!!!!!!!!!!!!!!!!!
			simple CPU parallel solver
			!!!!!!!!!!!!!!!!!!!!!!! */

			if (use_drhodt) {
				int previous_rho_idx = std::distance(n.fieldNames.begin(),
					std::find(n.fieldNames.begin(), n.fieldNames.end(), "previous_rho"));
				if (!n.isAllocated(previous_rho_idx)) {
					use_drhodt = false;
					std::cerr << "disabeling using drho/dt because 'previous_rho' isn't alocated !\n";
				}
			}

			// buffers
			std::vector<Float> rates(reactions.size());
			eigen::Vector<Float> RHS(dimension + 1), DY_T(dimension + 1), Y_buffer(dimension), Y(dimension);
			eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

			int num_threads;
			#pragma omp parallel
			#pragma omp master
			num_threads = omp_get_num_threads();
			int omp_batch_size = util::dynamic_batch_size(n_particles, num_threads);

			#pragma omp parallel for firstprivate(Y, Mp, RHS, DY_T, rates, Y_buffer, reactions/*, construct_rates_BE, eos*/) schedule(dynamic, omp_batch_size)
			for (size_t i = firstIndex; i < lastIndex; ++i)
				if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp) {
					// copy to local vector
					for (int j = 0; j < dimension; ++j)
						Y[j] = n.Y[j][i];

					// compute drho/dt
					Float drho_dt = 0;
					if (use_drhodt)
						n.previous_rho[i] = (n.rho[i] - n.previous_rho[i])/previous_dt;

					// solve
					nnet::solve_system_substep(dimension,
						Mp.data(), RHS.data(), DY_T.data(), rates.data(),
						reactions, construct_rates_BE, eos,
						Y.data(), n.temp[i], Y_buffer.data(),
						n.rho[i], drho_dt, hydro_dt, n.dt[i],
						jumpToNse);

					// copy from local vector
					for (int j = 0; j < dimension; ++j)
						n.Y[j][i] = Y[j];
				}
		}
	}


	/*! @brief function to copute the helmholtz eos
	 * 
	 * @param n           nuclearDataType including a field of nuclear abundances "Y"
     * @param firstIndex  first (included) particle considered in n
	 * @param lastIndex   last (excluded) particle considered in n
	 * @param Z           vector of number of charge (used in eos)
	 */
	template<class Data, class Vector>
	void computeHelmEOS(Data &n, size_t firstIndex, size_t lastIndex, const Vector &Z) {
		const int dimension = n.Y.size();
		using Float = typename std::remove_reference<decltype(n.cv[0])>::type;

		if constexpr (HaveGpu<typename Data::AcceleratorType>{} && COMPILE_DEVICE) {

			/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			simple GPU application of the eos
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if COMPILE_DEVICE
			// copy data to the GPU
			Float *Z_dev;
			gpuErrchk(cudaMalloc((void**)&Z_dev, dimension*sizeof(Float)));
			gpuErrchk(cudaMemcpy((void*)Z_dev, (void*)Z.data(), dimension*sizeof(Float), cudaMemcpyHostToDevice));

			// copy pointers to GPU
			std::vector<Float*> Y_raw_ptr(dimension);
			Float **Y_dev_ptr;
			gpuErrchk(cudaMalloc((void**)&Y_dev_ptr, dimension*sizeof(Float*)));
			for (int i = 0; i < dimension; ++i)
	            Y_raw_ptr[i] = (Float*)thrust::raw_pointer_cast(n.devData.Y[i].data() + firstIndex); // store Y raw pointer to CPU
	        gpuErrchk(cudaMemcpy((void*)Y_dev_ptr, (void*)Y_raw_ptr.data(), dimension*sizeof(Float*), cudaMemcpyHostToDevice));


			// call the cuda kernel wrapper
			cudaComputeHelmholtz(lastIndex - firstIndex, dimension, Z_dev,
				// read buffers:
		(Float*)thrust::raw_pointer_cast(n.devData.temp.data() + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.rho.data()  + firstIndex),
				Y_dev_ptr,
				// write buffers:
		(Float*)thrust::raw_pointer_cast(n.devData.u.data()    + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.cv.data()   + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.p.data()    + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.c.data()    + firstIndex),
		(Float*)thrust::raw_pointer_cast(n.devData.dpdT.data() + firstIndex));


			// debuging: check for error
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize()); 

			// free cuda buffer
			gpuErrchk(cudaFree((void*)Z_dev));
			gpuErrchk(cudaFree((void*)Y_dev_ptr));
#endif
		} else {

			/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			simple CPU parallel application of the eos
			!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
			std::vector<Float> Y(dimension);

			#pragma omp parallel for firstprivate(Y) schedule(static)
			for (size_t i = firstIndex; i < lastIndex; ++i) {
				// copy to local vector
				for (int j = 0; j < dimension; ++j)
					Y[j] = n.Y[j][i];

				// compute abar and zbar
				double abar = std::accumulate(Y.data(), Y.data() + dimension, (double)0.);
				double zbar =      eigen::dot(Y.data(), Y.data() + dimension, Z);

				auto eos_struct = nnet::eos::helmholtz(abar, zbar, n.temp[i], n.rho[i]);

				n.u[i]    = eos_struct.u;
				n.cv[i]   = eos_struct.cv;
				n.p[i]    = eos_struct.p;
				n.c[i]    = eos_struct.c;
				n.dpdT[i] = eos_struct.dpdT;
			}
		}
	}


	/*! @brief function initializing the partition of NuclearDataType using "maximal mixing"
	 * 
	 * @param firstIndex  first (included) particle considered in d
	 * @param lastIndex   last (excluded) particle considered in d
	 * @param d           ParticlesDataType (contains node_id and particle_id to be populated)
	 * @param n           nuclearDataType (contains mpi_partition to be populated)
	 */
	template<class ParticlesDataType,class nuclearDataType>
	void initializePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, nuclearDataType &n) {
#ifdef USE_MPI
		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);

		size_t n_particles = lastIndex - firstIndex;
		MPI_Allreduce(&n_particles, &n.numParticlesGlobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, n.comm);
#endif
	}
}