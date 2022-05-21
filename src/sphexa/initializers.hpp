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
	NuclearDataType<n_species, Float> initNuclearDataFromPos(ParticlesDataType &d, const initFunc initializer, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		NuclearDataType<n_species, Float> n;

		const size_t local_nuclear_n_particles = partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.rho, datatype, d.comm);

		// receiv position for initializer
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.x, x, datatype, d.comm);
		sphexa::mpi::directSyncDataFromPartition(partition, d.y, y, datatype, d.comm);
		sphexa::mpi::directSyncDataFromPartition(partition, d.z, z, datatype, d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromRadius(ParticlesDataType &d, const initFunc initializer, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		NuclearDataType<n_species, Float> n;

		const size_t local_nuclear_n_particles = partition.recv_partition.size();
		const size_t local_n_particles = d.x.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.rho, datatype, d.comm);

		// receiv position for initializer
		std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i)
			send_r[i] = std::sqrt(d.x[i]*d.x[i] + d.y[i]*d.y[i] + d.z[i]*d.z[i]);
		sphexa::mpi::directSyncDataFromPartition(partition, send_r, r, datatype, d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromRho(ParticlesDataType &d, const initFunc initializer, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		NuclearDataType<n_species, Float> n;

		const size_t local_nuclear_n_particles = partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.rho, datatype, d.comm);

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
	NuclearDataType<n_species, Float> initNuclearDataFromConst(ParticlesDataType &d, const Vector &Y0, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		NuclearDataType<n_species, Float> n;

		const size_t local_nuclear_n_particles = partition.recv_partition.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.rho, datatype, d.comm);

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = Y0;

		return n;
	}
}