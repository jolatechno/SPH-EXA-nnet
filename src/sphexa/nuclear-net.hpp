#pragma once

#include <numeric>
#include <omp.h>

#include "../eos/helmholtz.hpp"
#include "nuclear-data.hpp"

#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
#endif

#ifdef USE_CUDA
	#include "../eigen/cudaSolver.hpp"
#endif

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
#else
	template<class Dataset>
	auto sphexa::getOutputArrays(Dataset &dataset);
#endif

#include "../nuclear-net.hpp"

namespace sphexa::sphnnet {
	/// function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y.
	/**
	 * TODO
	 */
	template<class Data, class func_rate, class func_BE, class func_eos, typename Float, class nseFunction=void*>
	void computeNuclearReactions(Data &n, const Float hydro_dt, const Float previous_dt,
		const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const nseFunction jumpToNse=NULL)
	{
		const size_t n_particles = n.temp.size();
		const int dimension = n.Y[0].size();
		
#ifndef USE_CUDA
		/* !!!!!!!!!!!!!
		simple CPU parallel solver
		!!!!!!!!!!!!! */
		#pragma omp parallel
		{
			// buffers
			Float drho_dt;
			eigen::Vector<Float> RHS(dimension + 1), Y_buffer(dimension);
			eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

			#pragma omp for schedule(dynamic)
			for (size_t i = 0; i < n_particles; ++i) {
				drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

				if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp)
					nnet::solve_system_substep(Mp, RHS,
						reactions, construct_rates, construct_BE, eos,
						n.Y[i], n.temp[i], Y_buffer,
						n.rho[i], drho_dt, hydro_dt, n.dt[i],
						jumpToNse);
			}
#else
		/* !!!!!!!!!!!!!
		GPU batch solver
		!!!!!!!!!!!!! */
		// intitialized bash solver data
		eigen::cudasolver::batch_solver<Float> batch_solver(dimension + 1);

		int num_threads_still_burning = 0;
		#pragma omp parallel
		{
			// buffers
			Float drho_dt;
			eigen::Vector<Float> vect_buffer(dimension + 1);
			eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

			// get number of thread and thread number
			const int num_threads = omp_get_num_threads();
			const int thread_id   = omp_get_thread_num();

			// initialize limits
			const size_t particleBegin     = n_particles                  *thread_id      /num_threads;
			const size_t particleEnd       = n_particles                  *(thread_id + 1)/num_threads;
			const size_t batchBegin        = eigen::cudasolver::batch_size*thread_id      /num_threads;
			const size_t batchEnd          = eigen::cudasolver::batch_size*(thread_id + 1)/num_threads;
			const size_t batchParticleSize = particleEnd - particleBegin;

			// data for batch initialization
			std::vector<int>   iter(batchParticleSize, 0);
			std::vector<bool>  finished(batchParticleSize, false);
			std::vector<Float> elapsed_time(batchParticleSize, 0.);
			std::vector<Float> T_buffer(batchParticleSize);
			decltype(n.Y)      Y_buffer(batchParticleSize);

			// intialize buffers
			for (size_t i = particleBegin; i < particleEnd; ++i) {
				T_buffer[i - particleBegin] = n.temp[i];
				Y_buffer[i - particleBegin] = n.Y[i]; 
			}


			// solving loop
			bool still_burning = true;
			while (true) {
				// prepare system
				size_t batchID = batchBegin;
				if (still_burning)
					for (size_t i = particleBegin; i < particleEnd; ++i)
						if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp &&
							!finished[i - particleBegin])
						{
							drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

							// preparing system
							++iter[i - particleBegin];
							nnet::prepare_system_substep(Mp, vect_buffer,
								reactions, construct_rates, construct_BE, eos,
								n.Y[i], n.temp[i],
								Y_buffer[i - particleBegin], T_buffer[i - particleBegin],
								n.rho[i], drho_dt,
								hydro_dt, elapsed_time[i - particleBegin], n.dt[i],
								jumpToNse);

							// insert into batch solver
							batch_solver.insert_system(batchID, Mp.data(), vect_buffer.data());

							// break condition
							++batchID;
							if (batchID >= batchEnd)
								break;
						}

				//std::cout << batchID << "/" << batchEnd << " batch size for rank " << thread_id << "\n";

				// batch solves
				#pragma omp barrier
				#pragma omp master
				{
					batch_solver.solve();
					num_threads_still_burning = 0;
				}
				#pragma omp barrier

				// finalize
				batchID = batchBegin;
				size_t num_particle_still_burning = 0;
				if (still_burning)
					for (size_t i = particleBegin; i < particleEnd; ++i)
						if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp &&
							!finished[i - particleBegin])
						{

							// retrieve results
							batch_solver.get_res(batchID, vect_buffer.data());

							// finalize
							if(nnet::finalize_system_substep(
								n.Y[i], n.temp[i],
								Y_buffer[i - particleBegin], T_buffer[i - particleBegin],
								vect_buffer, hydro_dt, elapsed_time[i - particleBegin],
								n.dt[i], iter[i - particleBegin]))
							{
								finished[i - particleBegin] = true;
							} else
								++num_particle_still_burning;

							// break condition
							++batchID;
							if (batchID >= batchEnd)
								break;
						}

				// check if particles are still burning
				still_burning = (num_particle_still_burning > 0);

				// check the number of threads still burning
				#pragma omp atomic
				num_threads_still_burning += still_burning;

				// exit if no threads are still burning
				#pragma omp barrier
				if (num_threads_still_burning == 0)
					break;
			}
#endif
		}
	}

	/// function to copute the helmholtz eos
	/**
	 * TODO
	 */
	template<class Data, class Vector>
	void computeHelmEOS(Data &n, const Vector &Z) {
		size_t n_particles = n.Y.size();

		const nnet::eos::helmholtz helm(Z);

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n_particles; ++i) {
			auto eos_struct = helm(n.Y[i], n.temp[i], n.rho[i]);

			n.u[i] = eos_struct.u;
			n.c[i] = eos_struct.c;
			n.p[i] = eos_struct.p;
		}
	}

#ifdef USE_MPI
	/// function initializing the partition of NuclearDataType from 
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void initializePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float> &n) {
		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);
	}


	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void hydroToNuclearUpdate(ParticlesDataType &d, NuclearDataType<n_species, Float> &n, const std::vector<std::string> &sync_fields) {
		std::vector<int>         outputFieldIndicesNuclear = n.outputFieldIndices, outputFieldIndicesHydro = d.outputFieldIndices;
		std::vector<std::string> outputFieldNamesNuclear   = n.outputFieldNames,   outputFieldNamesHydro   = d.outputFieldNames;

		// get particle data
		n.setOutputFields(sync_fields);
		auto nuclearData = sphexa::getOutputArrays(n);

		// get nuclear data
		std::vector<std::string> particleDataFields = sync_fields;
		std::replace(particleDataFields.begin(), particleDataFields.end(), std::string("previous_rho"), std::string("rho")); // replace "previous_rho" by "rho" for particle data
		d.setOutputFields(particleDataFields);
		auto particleData  = sphexa::getOutputArrays(d);

		const int n_fields = sync_fields.size();
		if (particleData.size() != n_fields || nuclearData.size() != n_fields)
			throw std::runtime_error("Not the right number of fields to synchronize in sendHydroData !\n");

		for (int field = 0; field < n_fields; ++field)
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::directSyncDataFromPartition(n.partition, send, recv, d.comm);
				}, particleData[field], nuclearData[field]);

		n.outputFieldIndices = outputFieldIndicesNuclear, d.outputFieldIndices = outputFieldIndicesHydro;
		n.outputFieldNames   = outputFieldNamesNuclear,   d.outputFieldNames   = outputFieldNamesHydro;
	}

	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void nuclearToHydroUpdate(ParticlesDataType &d, NuclearDataType<n_species, Float> &n, const std::vector<std::string> &sync_fields) {
		std::vector<int>         outputFieldIndicesNuclear = n.outputFieldIndices, outputFieldIndicesHydro = d.outputFieldIndices;
		std::vector<std::string> outputFieldNamesNuclear   = n.outputFieldNames,   outputFieldNamesHydro   = d.outputFieldNames;

		d.setOutputFields(sync_fields);
		n.setOutputFields(sync_fields);

		using FieldType = std::variant<float*, double*, int*, unsigned*, size_t*, uint8_t*/*bool* */>;

		std::vector<FieldType> particleData = sphexa::getOutputArrays(d);
		std::vector<FieldType> nuclearData  = sphexa::getOutputArrays(n);

		const int n_fields = sync_fields.size();
		if (particleData.size() != n_fields || nuclearData.size() != n_fields)
			throw std::runtime_error("Not the right number of fields to synchronize in recvHydroData !\n");

		for (int field = 0; field < n_fields; ++field)
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::reversedSyncDataFromPartition(n.partition, send, recv, d.comm);
				}, nuclearData[field], particleData[field]);

		n.outputFieldIndices = outputFieldIndicesNuclear, d.outputFieldIndices = outputFieldIndicesHydro;
		n.outputFieldNames   = outputFieldNamesNuclear,   d.outputFieldNames   = outputFieldNamesHydro;
	}
#endif
}