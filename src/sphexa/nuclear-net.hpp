#pragma once

#include <numeric>

#include "../eos/helmholtz.hpp"
#include "nuclear-data.hpp"

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
	template<class Data, class func_rate, class func_BE, class func_eos, typename Float>
	void computeNuclearReactions(Data &n, const Float hydro_dt, const Float previous_dt,
		const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {
		const size_t n_particles = n.temp.size();

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n_particles; ++i)
			if (n.burning[i]) {
				Float drho_dt = (n.rho[i] - n.previous_rho[i])/previous_dt;

				std::tie(n.Y[i], n.temp[i]) = nnet::solve_system_substep(reactions, construct_rates, construct_BE, eos,
					n.Y[i], n.temp[i],
					n.rho[i], drho_dt, hydro_dt, n.dt[i]);
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
	void initializePartition(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float> &n, Float min_temp=nnet::constants::min_temp, Float min_rho=nnet::constants::min_rho) {
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = firstIndex; i < lastIndex; ++i)
			d.burning[i] = d.rho[i] > min_rho && d.temp[i] > min_temp;

		n.partition = sphexa::mpi::partitionFromPointers(firstIndex, lastIndex, d.node_id, d.particle_id, d.burning, n.burning, d.comm);
	}


	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void hydroToNuclearUpdate(ParticlesDataType &d, NuclearDataType<n_species, Float> &n, const std::vector<std::string> &sync_fields) {
		if (!n.first_step)
			std::swap(n.rho, n.previous_rho);

		std::vector<int>         outputFieldIndicesNuclear = n.outputFieldIndices, outputFieldIndicesHydro = d.outputFieldIndices;
		std::vector<std::string> outputFieldNamesNuclear   = n.outputFieldNames,   outputFieldNamesHydro   = d.outputFieldNames;

		d.setOutputFields(sync_fields);
		n.setOutputFields(sync_fields);

		using FieldType = std::variant<float*, double*, int*, unsigned*, size_t*, uint8_t*/*bool* */>;

		std::vector<FieldType> particleData = sphexa::getOutputArrays(d);
		std::vector<FieldType> nuclearData  = sphexa::getOutputArrays(n);

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

		if (n.first_step) {
			std::copy(n.rho.begin(), n.rho.end(), n.previous_rho.begin());
			n.first_step = false;
		}
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