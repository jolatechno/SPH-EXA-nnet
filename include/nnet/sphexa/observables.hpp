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

#include "../eigen/eigen.hpp"
#include "util/algorithm.hpp"

#include "mpi/mpi-wrapper.hpp"

#include "sph/data_util.hpp"

namespace sphexa::sphnnet {
	/*! @brief function to compute the total nuclear energy
	 * 
	 * @param n   nuclearDataType containing a list of magnitude (named Y, being a vector of array)
	 * @param BE  binding energy vector used to compute nuclear energy
	 * 
	 * Returns the total nuclear binding energy (negative).
	 */
	template<class Data, typename Float>
	Float totalNuclearEnergy(Data const &n, const Float *BE) {
		const size_t n_particles = n.temp.size();
		const int dimension = n.Y[0].size();

#if COMPILE_DEVICE
		if constexpr (HaveGpu<typename Data::AcceleratorType>{} && false /* NOT IMPLEMENTED */) {

			/* TODO */

			return 0.0;
		}
#endif
		Float total_energy = 0;
		#pragma omp parallel for schedule(static) reduction(+:total_energy)
		for (size_t i = 0; i < n_particles; ++i)
			total_energy += -eigen::dot(n.Y[i].data(), n.Y[i].data() + dimension, BE)*n.m[i];

#ifdef USE_MPI
		double mpi_buffer_total_energy = (double)total_energy;
		MPI_Allreduce(MPI_IN_PLACE, &mpi_buffer_total_energy, 1, MPI_DOUBLE, MPI_SUM, n.comm);
		total_energy = (Float)mpi_buffer_total_energy;
#endif

		return total_energy;
	}
}