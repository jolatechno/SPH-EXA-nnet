#pragma once

#include "mpi-wrapper.hpp"
#include "nuclear-data.hpp"

#include <numeric>

#include "../../src/nuclear-net.hpp"

namespace sphexa::sphnnet {
	/// function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y.
	/**
	 * TODO
	 */
	template<class Data, class func_rate, class func_BE, class func_eos, typename Float>
	void compute_nuclear_reactions(Data &n, const Float hydro_dt,
		const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {
		const size_t n_particles = n.T.size();

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n_particles; ++i) {
			Float drho_dt = 0.; (n.rho[i] - n.previous_rho[i])/hydro_dt;

			std::tie(n.Y[i], n.T[i]) = nnet::solve_system_superstep(reactions, construct_rates, construct_BE, eos,
				n.Y[i], n.T[i], n.rho[i], /*n.drho_dt[i]*/drho_dt, hydro_dt, n.dt[i]);
		} 

	}

	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species,typename Float=double>
	void sendHydroData(const ParticlesDataType &d, NuclearDataType<n_species, Float> &n, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		std::swap(n.rho, n.previous_rho);

		sphexa::mpi::direct_sync_data_from_partition(partition, d.rho, n.rho, datatype);
		sphexa::mpi::direct_sync_data_from_partition(partition, d.T,   n.T,   datatype);
	}

	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species,typename Float=double>
	void recvHydroData(ParticlesDataType &d, const NuclearDataType<n_species, Float> &n, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		sphexa::mpi::reversed_sync_data_from_partition(partition, n.T, d.T, datatype);
	}

	/// intialize nuclear data, from a function of positions:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class initFunc, class ParticlesDataType>
	NuclearDataType<n_species, Float> initNuclearData(ParticlesDataType &d, const initFunc initializer, MPI_Datatype datatype) {
		int rank, size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		NuclearDataType<n_species, Float> n;

		// share the number of particle per node
		size_t local_n_particles = d.T.size();
		std::vector<size_t> n_particles(size);
		MPI_Allgather(&local_n_particles, 1, MPI_UNSIGNED_LONG_LONG,
					  &n_particles[0],    1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		// initialize "node_id" and "particle_id"
		size_t global_idx_begin = std::accumulate(n_particles.begin(), n_particles.begin() + rank, (size_t)0);
		for (size_t i = 0; i < local_n_particles; ++i) {
			size_t global_idx = global_idx_begin + i;
			d.node_id[i] = global_idx % size;
			d.particle_id[i] = global_idx / size;
		}

		// initialize partition
		sphexa::mpi::mpi_partition partition = sphexa::mpi::partition_from_pointers(d.node_id, d.particle_id);
		size_t local_nuclear_n_particles = partition.recv_disp[size];

		// receiv position for initializer
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::direct_sync_data_from_partition(partition, d.x, x, datatype);
		sphexa::mpi::direct_sync_data_from_partition(partition, d.y, y, datatype);
		sphexa::mpi::direct_sync_data_from_partition(partition, d.z, z, datatype);

		// intialize nuclear data
		n.resize(local_nuclear_n_particles);
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < local_nuclear_n_particles; ++i)
			n.Y[i] = initializer(x[i], y[i], z[i]);

		// share the initial rho
		sphexa::mpi::direct_sync_data_from_partition(partition, d.rho, n.rho, datatype);

		return n;
	}
}