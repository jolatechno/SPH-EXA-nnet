#pragma once

#include <numeric>
#include <omp.h>

#include "nuclear-io.hpp"
#include "nuclear-data.hpp"
#include "../eos/helmholtz.hpp"

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

	#include "CUDA/nuclear-net.cu.cuh"
	#include "../CUDA/nuclear-net.hpp"
#endif

#include "../nuclear-net.hpp"

namespace sphexa::sphnnet {
	/// function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y.
	/**
	 * TODO
	 */
	template<class Data, class func_type, class func_eos, typename Float, class nseFunction=void*>
	void computeNuclearReactions(Data &n, const Float hydro_dt, const Float previous_dt,
		const nnet::reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		const nseFunction jumpToNse=NULL)
	{
		n.minDt_m1 = n.minDt;
		n.minDt    = hydro_dt;
		n.ttot    += n.minDt;
		++n.iteration;

		const size_t n_particles = n.temp.size();
		const int dimension = n.Y[0].size();
		
#ifdef USE_CUDA
		if constexpr (HaveGpu<typename Data::AcceleratorType>{}) {
			/* !!!!!!!!!!!!!
			GPU non-batch solver
			!!!!!!!!!!!!! */
			cudaComputeNuclearReactions(n_particles, dimension,
				thrust::raw_pointer_cast(n.rho.data()),
				thrust::raw_pointer_cast(n.previous_rho.data()),
		(Float*)thrust::raw_pointer_cast(n.Y.data()),
				thrust::raw_pointer_cast(n.temp.data()),
				thrust::raw_pointer_cast(n.dt.data()),
				hydro_dt, previous_dt,
				nnet::move_to_gpu(reactions), construct_rates_BE, eos);

			/* debuging: check for error */
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());

			return;
		}
#endif
		/* !!!!!!!!!!!!!!!!!!!!!!!
		simple CPU parallel solver
		!!!!!!!!!!!!!!!!!!!!!!! */

		// buffers
		std::vector<Float> rates(reactions.size()), drates_dT(reactions.size());
		eigen::Vector<Float> RHS(dimension + 1), DY_T(dimension + 1), Y_buffer(dimension);
		eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

		int num_threads;
		#pragma omp parallel
		#pragma omp master
		num_threads = omp_get_num_threads();
		int omp_batch_size = util::dynamic_batch_size(n_particles, num_threads);

		#pragma omp parallel for firstprivate(Mp, RHS, DY_T, rates, drates_dT, Y_buffer, reactions/*, construct_rates_BE, eos*/) schedule(dynamic, omp_batch_size)
		for (size_t i = 0; i < n_particles; ++i) 
			if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp) {
				// compute drho/dt
				Float drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

				// solve
				nnet::solve_system_substep(dimension,
					Mp.data(), RHS.data(), DY_T.data(), rates.data(), drates_dT.data(),
					reactions, construct_rates_BE, eos,
					n.Y[i].data(), n.temp[i], Y_buffer.data(),
					n.rho[i], drho_dt, hydro_dt, n.dt[i],
					jumpToNse);
			}
	}

	/// function to copute the helmholtz eos
	/**
	 * TODO
	 */
	template<class Data, class Vector>
	void computeHelmEOS(Data &n, const Vector &Z) {
		size_t n_particles = n.Y.size();

		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i) {
			// compute abar and zbar
			double abar = std::accumulate(n.Y[i].begin(), n.Y[i].end(), (double)0.);
			double zbar = eigen::dot(n.Y[i].begin(), n.Y[i].end(), Z);

			auto eos_struct = nnet::eos::helmholtz(abar, zbar, n.temp[i], n.rho[i]);

		 // n.u[i]  = eos_struct.u;
			n.cv[i] = eos_struct.cv;
			n.c[i]  = eos_struct.c;
			n.p[i]  = eos_struct.p;
		}
	}

	/// function initializing the partition of NuclearDataType from 
	/**
	 * TODO
	 */
	template<class ParticlesDataType,class nuclearDataType, typename Float=double>
	void initializePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, nuclearDataType &n) {
#ifdef USE_MPI
		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);

		size_t n_particles = lastIndex - firstIndex;
		MPI_Allreduce(&n_particles, &n.numParticlesGlobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, n.comm);
#endif
	}


	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void hydroToNuclearUpdate(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
#ifdef USE_MPI
		// get data
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			if (field == "previous_rho")
				field = "rho";
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			// send
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::directSyncDataFromPartition(n.partition, send->data(), recv->data(), d.comm);
				}, particleData[particleFieldIdx], nuclearData[nuclearFieldIdx]);
		}
#endif
	}

	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void nuclearToHydroUpdate(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
#ifdef USE_MPI
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::reversedSyncDataFromPartition(n.partition, send->data(), recv->data(), d.comm);
				}, nuclearData[nuclearFieldIdx], particleData[particleFieldIdx]);
		}
#endif
	}
}