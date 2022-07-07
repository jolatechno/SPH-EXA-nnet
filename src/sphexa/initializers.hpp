#pragma once

#include <numeric>

#include "nuclear-data.hpp"
#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
#endif

namespace sphexa::sphnnet {
	/// intialize nuclear data, from a function of positions:
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromPos(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		

		// receiv position for initializer
#ifdef USE_MPI
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.x.data(), x.data(), d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.y.data(), y.data(), d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.z.data(), z.data(), d.comm);
#else
		std::vector<Float> &x = d.x, &y = d.y, &z = d.z;
#endif

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(x[i], y[i], z[i]);
	}

	/// intialize nuclear data, from a function of radius:
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromRadius(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif
		const size_t local_n_particles = d.x.size();

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// receiv position for initializer
#ifdef USE_MPI
		std::vector<Float> send_r(local_n_particles), r(local_nuclear_n_particles, d.comm);
#else
		std::vector<Float> r(local_n_particles), &send_r = r;
#endif
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i)
			send_r[i] = std::sqrt(d.x[i]*d.x[i] + d.y[i]*d.y[i] + d.z[i]*d.z[i]);
#ifdef USE_MPI
		sphexa::mpi::directSyncDataFromPartition(n.partition, send_r.data(), r.data(), d.comm);
#endif

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(r[i]);
	}

	/// intialize nuclear data, from a function of density:
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float, class AccType, class initFunc, class ParticlesDataType>
	void initNuclearDataFromRho(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, AccType> &n, const initFunc initializer) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);
#ifdef USE_MPI
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho.data(), n.rho.data(), d.comm);
#else
		n.rho = d.rho;
#endif

		// intialize nuclear data
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(n.rho[i]);
	}


		/// intialize nuclear data as a constant:
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float, class AccType, class Vector, class ParticlesDataType>
	void initNuclearDataFromConst(size_t firstIndex, size_t lastIndex, ParticlesDataType &d, NuclearDataType<n_species, Float, AccType> &n, const Vector &Y0) {
#ifdef USE_MPI
		int size;
		MPI_Comm_size(d.comm, &size);

		n.comm = d.comm;
		sphexa::sphnnet::initializePartition(firstIndex, lastIndex, d, n);
		const size_t local_nuclear_n_particles = n.partition.recv_disp[size];
#else
		const size_t local_nuclear_n_particles = d.x.size();
#endif

		// share the initial rho
		n.resize(local_nuclear_n_particles);

		// intialize nuclear data
		const int dimension = n.Y[0].size();
		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_nuclear_n_particles; ++i)
			for (int j = 0; j < dimension; ++j)
				n.Y[i][j] = Y0[j];
	}
}