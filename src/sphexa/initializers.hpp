#pragma once

#include <numeric>

#include "nuclear-data.hpp"
#include "mpi/mpi-wrapper.hpp"

namespace sphexa::sphnnet {
	/// intialize nuclear data, from a function of positions:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class initFunc, class ParticlesDataType>
	NuclearDataType<n_species, Float> initNuclearDataFromPos(ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		sphexa::sphnnet::initializePartition(d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho, n.rho, d.comm);

		// receiv position for initializer
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.x, x, d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.y, y, d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.z, z, d.comm);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(x[i], y[i], z[i]);

		return n;
	}

	/// intialize nuclear data, from a function of radius:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class initFunc, class ParticlesDataType>
	NuclearDataType<n_species, Float> initNuclearDataFromRadius(ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		sphexa::sphnnet::initializePartition(d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_partition.size();
		const size_t local_n_particles = d.x.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho, n.rho, d.comm);

		// receiv position for initializer
		std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i)
			send_r[i] = std::sqrt(d.x[i]*d.x[i] + d.y[i]*d.y[i] + d.z[i]*d.z[i]);
		sphexa::mpi::directSyncDataFromPartition(n.partition, send_r, r, d.comm);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(r[i]);

		return n;
	}

	/// intialize nuclear data, from a function of density:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class initFunc, class ParticlesDataType>
	NuclearDataType<n_species, Float> initNuclearDataFromRho(ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		sphexa::sphnnet::initializePartition(d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho, n.rho, d.comm);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(n.rho[i]);

		return n;
	}


		/// intialize nuclear data as a constant:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class Vector, class ParticlesDataType>
	NuclearDataType<n_species, Float> initNuclearDataFromConst(ParticlesDataType &d, const Vector &Y0) {
		NuclearDataType<n_species, Float> n;

		sphexa::sphnnet::initializePartition(d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho, n.rho, d.comm);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = Y0;

		return n;
	}
}