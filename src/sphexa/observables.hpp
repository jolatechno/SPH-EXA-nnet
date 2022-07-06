#pragma once

#include <numeric>
#include <omp.h>

#include "../eigen/eigen.hpp"
#include "util/algorithm.hpp"

#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
#endif

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
#endif

#ifdef USE_CUDA
	#include <device_launch_parameters.h>
	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include <cuda_runtime.h>

	#include <thrust/device_vector.h>

	#include "../CUDA/nuclear-net.hpp"
	// #include "CUDA/nuclear-net.cuh"
	#include "CUDA/nuclear-net.cu"
#endif

namespace sphexa::sphnnet {
	/// function to compute the total nuclear energy
	/**
	 * TODO
	 */
	template<class Data, typename Float>
	Float totalNuclearEnergy(Data const &n, const Float *BE) {
		const size_t n_particles = n.temp.size();
		const int dimension = n.Y[0].size();

#ifdef USE_CUDA
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