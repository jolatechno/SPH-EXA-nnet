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
	NuclearDataType<n_species, Float> initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n, 0., 0.);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size*2];

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		

		// receiv position for initializer
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.x.data(), x.data(), d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.y.data(), y.data(), d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.z.data(), z.data(), d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n, 0., 0.);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size*2];
		const size_t local_n_particles = d.x.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// receiv position for initializer
		std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i)
			send_r[i] = std::sqrt(d.x[i]*d.x[i] + d.y[i]*d.y[i] + d.z[i]*d.z[i]);
		sphexa::mpi::directSyncDataFromPartition(n.partition, send_r.data(), r.data(), d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const initFunc initializer) {
		NuclearDataType<n_species, Float> n;

		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n, 0., 0.);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size*2];

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho.data(), n.rho.data(), d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, const Vector &Y0) {
		NuclearDataType<n_species, Float> n;

		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n, 0., 0.);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size*2];

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = Y0;

		return n;
	}
}