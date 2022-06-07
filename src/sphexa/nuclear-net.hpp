#pragma once

#include <numeric>
#include <omp.h>

#include "nuclear-io.hpp"
#include "nuclear-data.hpp"
#include "../eos/helmholtz.hpp"

#include "../eigen/eigen.hpp"
#include "../eigen/batchSolver.hpp"

#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
#endif

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
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
			std::vector<Float> rates(reactions.size()), drates_dT(reactions.size());
			eigen::Vector<Float> RHS(dimension + 1), Y_buffer(dimension);
			eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

			#pragma omp for schedule(dynamic)
			for (size_t i = 0; i < n_particles; ++i) 
				if (n.rho[i] > nnet::constants::min_rho && n.temp[i] > nnet::constants::min_temp) {
					// compute drho/dt
					drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

					// solve
					nnet::solve_system_substep(dimension,
						Mp.data(), RHS.data(), rates.data(), drates_dT.data(),
						reactions, construct_rates, construct_BE, eos,
						n.Y[i].data(), n.temp[i], Y_buffer.data(),
						n.rho[i], drho_dt, hydro_dt, n.dt[i],
						jumpToNse);
				}
		}
#else
		/* !!!!!!!!!!!!!
		GPU batch solver
		!!!!!!!!!!!!! */
		// intitialized bash solver data
		const int numDevice = eigen::batchSolver::util::getNumDevice();
		size_t batch_size = std::min(n_particles, eigen::batchSolver::constants::max_batch_size*numDevice);

#ifdef USE_CUDA
		eigen::batchSolver::CUDAsolver<Float> batch_solver(batch_size, dimension + 1);
#else
		eigen::batchSolver::CPUsolver<Float> batch_solver(batch_size, dimension + 1);
#endif

		// data for batch initialization
		std::vector<int>              iter(n_particles, 1);
		std::vector<uint8_t/*bool*/>  burning(n_particles, true);
		std::vector<size_t>           particle_ids(batch_size);
		std::vector<Float>            elapsed_time(n_particles, 0.);
		std::vector<Float>            temp_buffers(n_particles);
		std::vector<Float>            Y_buffers(n_particles*dimension);

		// solving loop
		while (true) {
			// compute particle_ids
			size_t batchID = 0;
			for (size_t i = 0; i < n_particles && batchID < batch_size; ++i) 
				if (burning[i])
					if (n.rho[i] < nnet::constants::min_rho || n.temp[i] < nnet::constants::min_temp) {
						burning[i] = false;
					} else {
						particle_ids[batchID] = i;
						++batchID;
					}
			batch_size = batchID;


			// fall back to simpler CPU non-batch solver if the number of particle still burning is low enough
			if (batch_size <= eigen::batchSolver::constants::min_batch_size
			// or if no devices are available
				|| numDevice == 0)
			{
				#pragma omp parallel
				{
					// buffers
					Float drho_dt;
					std::vector<Float> rates(reactions.size()), drates_dT(reactions.size());
					eigen::Vector<Float> RHS(dimension + 1), Y_buffer(dimension);
					eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

					#pragma omp for schedule(dynamic)
					for (size_t i = 0; i < n_particles; ++i)
						if (burning[i]) {
							// compute drho/dt
							drho_dt = n.previous_rho[i] <= 0 ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

							// compute the remaining time step to integrate over
							Float elapsed_time_ = elapsed_time[i], temp_buffer = temp_buffers[i];
							for (int j = 0; j < dimension; ++j)
								Y_buffer[j] = Y_buffers[i*dimension + j];

							// solve
							for (int j = iter[i];; ++j) {
								// generate system
								nnet::prepare_system_substep(dimension,
									Mp.data(), RHS.data(), rates.data(), drates_dT.data(),
									reactions, construct_rates, construct_BE, eos,
									n.Y[i].data(), n.temp[i],
									Y_buffer.data(), temp_buffer,
									n.rho[i], drho_dt,
									hydro_dt, elapsed_time_, n.dt[i], j,
									jumpToNse);

							// solve M*D{T, Y} = RHS
							auto DY_T = eigen::solve(Mp.data(), RHS.data(), dimension + 1, nnet::constants::epsilon_system);

							// finalize
							if(nnet::finalize_system_substep(dimension,
								n.Y[i].data(), n.temp[i],
								Y_buffer.data(), temp_buffer,
								DY_T.data(), hydro_dt, elapsed_time_,
								n.dt[i], j))
							{
								break;
							}
						}
					}
				}

				// leaving
				break;
			}



			// prepare system
			#pragma omp parallel
			{
				// buffers
				std::vector<Float> rates(reactions.size()), drates_dT(reactions.size());
				eigen::Vector<Float> RHS(dimension + 1);
				eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

				#pragma omp for schedule(dynamic)
				for (size_t batchID = 0; batchID < batch_size; ++batchID) {
					size_t i = particle_ids[batchID];

					// compute drho/dt
					Float drho_dt = n.previous_rho[i] <= 0. ? 0. : (n.rho[i] - n.previous_rho[i])/previous_dt;

					// preparing system
					nnet::prepare_system_substep(dimension,
						Mp.data(), RHS.data(), rates.data(), drates_dT.data(),
						reactions, construct_rates, construct_BE, eos,
						n.Y[i].data(), n.temp[i],
						&Y_buffers[i*dimension], temp_buffers[i],
						n.rho[i], drho_dt,
						hydro_dt, elapsed_time[i], n.dt[i], iter[i],
						jumpToNse);

					// insert
					auto [Mp_batch, RHS_batch] = batch_solver.get_system_reference(batchID);
					std::copy(RHS.data(), RHS.data() + (dimension + 1),                 RHS_batch);
					std::copy(Mp.data(),  Mp.data()  + (dimension + 1)*(dimension + 1), Mp_batch);
				}
			}



			// solve
			batch_solver.solve(batch_size);



			// finalize
			#pragma omp parallel for schedule(dynamic)
			for (size_t batchID = 0; batchID < batch_size; ++batchID) {
				size_t i = particle_ids[batchID];
			
				// retrieve results
				auto res_buffer = batch_solver.get_res(batchID);

				// finalize
				if(nnet::finalize_system_substep(dimension,
					n.Y[i].data(), n.temp[i],
					&Y_buffers[i*dimension], temp_buffers[i],
					res_buffer, hydro_dt, elapsed_time[i],
					n.dt[i], iter[i]))
				{
					burning[i] = false;
				}

				// incrementing number of iteration
				++iter[i];
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

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n_particles; ++i) {
			// compute abar and zbar
			auto abar = std::accumulate(n.Y[i].begin(), n.Y[i].end(), 0.f);
			auto zbar = eigen::dot(n.Y[i].begin(), n.Y[i].end(), Z);

			auto eos_struct = nnet::eos::helmholtz(abar, zbar, n.temp[i], n.rho[i]);

		 // n.u[i]  = eos_struct.u;
			n.cv[i] = eos_struct.cv;
			n.c[i]  = eos_struct.c;
			n.p[i]  = eos_struct.p;
		}
	}

#ifdef USE_MPI
	/// function initializing the partition of NuclearDataType from 
	/**
	 * TODO
	 */
	template<class ParticlesDataType,class nuclearDataType, typename Float=double>
	void initializePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, nuclearDataType &n) {
		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.comm);
	}


	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void hydroToNuclearUpdate(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		std::vector<int>         outputFieldIndicesNuclear = n.outputFieldIndices, outputFieldIndicesHydro = d.outputFieldIndices, nuclearIOoutputFieldIndices = io::outputFieldIndices;
		std::vector<std::string> outputFieldNamesNuclear   = n.outputFieldNames,   outputFieldNamesHydro   = d.outputFieldNames,   nuclearIOoutputFieldNames   = io::outputFieldNames;

		// get particle data
		n.setOutputFields(sync_fields);
		auto nuclearData  = sphexa::getOutputArrays(n);

		// get nuclear data
		std::vector<std::string> particleDataFields = sync_fields;
		std::replace(particleDataFields.begin(), particleDataFields.end(), std::string("previous_rho"), std::string("rho")); // replace "previous_rho" by "rho" for particle data
		d.setOutputFields(particleDataFields);
		auto particleData = sphexa::getOutputArrays(d);

		const int n_fields = sync_fields.size();
		if (particleData.size() != n_fields || nuclearData.size() != n_fields)
			throw std::runtime_error("Not the right number of fields to synchronize in sendHydroData !\n");

		for (int field = 0; field < n_fields; ++field)
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::directSyncDataFromPartition(n.partition, send, recv, d.comm);
				}, particleData[field], nuclearData[field]);

		n.outputFieldIndices = outputFieldIndicesNuclear, d.outputFieldIndices = outputFieldIndicesHydro; io::outputFieldIndices = nuclearIOoutputFieldIndices;
		n.outputFieldNames   = outputFieldNamesNuclear,   d.outputFieldNames   = outputFieldNamesHydro;   io::outputFieldNames   = nuclearIOoutputFieldNames;
	}

	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void nuclearToHydroUpdate(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		std::vector<int>         outputFieldIndicesNuclear = n.outputFieldIndices, outputFieldIndicesHydro = d.outputFieldIndices, nuclearIOoutputFieldIndices = io::outputFieldIndices;
		std::vector<std::string> outputFieldNamesNuclear   = n.outputFieldNames,   outputFieldNamesHydro   = d.outputFieldNames,   nuclearIOoutputFieldNames   = io::outputFieldNames;

		d.setOutputFields(sync_fields);
		n.setOutputFields(sync_fields);

		auto particleData = sphexa::getOutputArrays(d);
		auto nuclearData  = sphexa::getOutputArrays(n);

		const int n_fields = sync_fields.size();
		if (particleData.size() != n_fields || nuclearData.size() != n_fields)
			throw std::runtime_error("Not the right number of fields to synchronize in recvHydroData !\n");

		for (int field = 0; field < n_fields; ++field)
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
					sphexa::mpi::reversedSyncDataFromPartition(n.partition, send, recv, d.comm);
				}, nuclearData[field], particleData[field]);

		n.outputFieldIndices = outputFieldIndicesNuclear, d.outputFieldIndices = outputFieldIndicesHydro; io::outputFieldIndices = nuclearIOoutputFieldIndices;
		n.outputFieldNames   = outputFieldNamesNuclear,   d.outputFieldNames   = outputFieldNamesHydro;   io::outputFieldNames   = nuclearIOoutputFieldNames;
	}
#endif
}