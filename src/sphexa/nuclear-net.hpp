#pragma once

#include <numeric>
#include <omp.h>

#include "../eos/helmholtz.hpp"
#include "nuclear-data.hpp"
#include "../eigen/batchSolver.hpp"

#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
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
		
#if !defined(USE_CUDA) && !defined(CPU_BATCH_SOLVER)
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
			for (size_t i = 0; i < n_particles; ++i) 
				if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp) {
					// compute drho/dt
					drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

					// solve
					nnet::solve_system_substep(Mp, RHS,
						reactions, construct_rates, construct_BE, eos,
						n.Y[i], n.temp[i], Y_buffer,
						n.rho[i], drho_dt, hydro_dt, n.dt[i],
						jumpToNse);
				}
		}
#else
		/* !!!!!!!!!!!!!
		GPU batch solver
		!!!!!!!!!!!!! */
		// intitialized bash solver data
		size_t batch_size = std::min(n_particles, eigen::batchSolver::constants::max_batch_size);
		eigen::batchSolver::batch_solver<Float> batch_solver(batch_size, dimension + 1);

		// data for batch initialization
		std::vector<int>              iter(n_particles, 0);
		std::vector<uint8_t/*bool*/>  burning(n_particles, true);
		std::vector<size_t>           batchIDs(n_particles + 1);
		std::vector<Float>            elapsed_time(n_particles, 0.);
		std::vector<Float>            T_buffer(n_particles);
		decltype(n.Y)                 Y_buffer(n_particles);

		// intialize buffers
		for (size_t i = 0; i < n_particles; ++i) {
			T_buffer[i] = n.temp[i];
			Y_buffer[i] = n.Y[i];
		}

		// solving loop
		while (true) {
			// update burning
			#pragma omp parallel for schedule(dynamic)
			for (size_t i = 0; i < n_particles; ++i) 
				if (n.rho[i] < nnet::constants::min_rho || n.temp[i] < nnet::constants::min_temp)
					burning[i] = false;

			// compute batchIDs
			batchIDs[0] = 0;
			__gnu_parallel::partial_sum(burning.begin(), burning.end(), batchIDs.begin() + 1);
			size_t num_particle_still_burning = batchIDs.back();

			// exit if no particles are burning
			if (num_particle_still_burning == 0)
				break;




			// fall back to simpler CPU non-batch solver if the number of particle still burning is low enough
			if (num_particle_still_burning < eigen::batchSolver::constants::min_batch_size) {
				#pragma omp parallel
				{
					// buffers
					Float drho_dt;
					eigen::Vector<Float> RHS(dimension + 1), Y_buffer(dimension);
					eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

					#pragma omp for schedule(dynamic)
					for (size_t i = 0; i < n_particles; ++i)
						if (burning[i]) {
							// compute drho/dt
							drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

							// compute the remaining time step to integrate over
							Float remaining_time = hydro_dt - elapsed_time[i];

							// solve
							nnet::solve_system_substep(Mp, RHS,
								reactions, construct_rates, construct_BE, eos,
								n.Y[i], n.temp[i], Y_buffer,
								n.rho[i], drho_dt, remaining_time, n.dt[i],
								jumpToNse);
						}
				}

				// living
				break;
			}




			// prepare system
			#pragma omp parallel
			{
				size_t batchID;
				Float drho_dt;
				eigen::Vector<Float> RHS(dimension + 1);
				eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

				#pragma omp for schedule(dynamic)
				for (size_t i = 0; i < n_particles; ++i)
					if (burning[i] && (batchID = batchIDs[i]) < eigen::batchSolver::constants::max_batch_size) {
						// compute drho/dt
						drho_dt = n.previous_rho[i] <= 0. ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

						// preparing system
						nnet::prepare_system_substep(Mp, RHS,
							reactions, construct_rates, construct_BE, eos,
							n.Y[i], n.temp[i],
							Y_buffer[i], T_buffer[i],
							n.rho[i], drho_dt,
							hydro_dt, elapsed_time[i], n.dt[i],
							jumpToNse);

						// insert into batch solver
						batch_solver.insert_system(batchID, Mp.data(), RHS.data());
					}
			}



			// solve
			size_t n_solve = std::min(num_particle_still_burning, eigen::batchSolver::constants::max_batch_size);
			batch_solver.solve(n_solve);



			// finalize
			#pragma omp parallel
			{
				size_t batchID;
				eigen::Vector<Float> res_buffer(dimension + 1);

				#pragma omp for schedule(dynamic)
				for (size_t i = 0; i < n_particles; ++i)
					if (burning[i] && (batchID = batchIDs[i]) < eigen::batchSolver::constants::max_batch_size) {
						// retrieve results
						batch_solver.get_res(batchID, res_buffer.data());

						// finalize
						if(nnet::finalize_system_substep(
							n.Y[i], n.temp[i],
							Y_buffer[i], T_buffer[i],
							res_buffer, hydro_dt, elapsed_time[i],
							n.dt[i], iter[i]))
						{
							burning[i] = false;
						}

						// incrementing number of iteration
						++iter[i];
					}
			}
		}
#endif
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