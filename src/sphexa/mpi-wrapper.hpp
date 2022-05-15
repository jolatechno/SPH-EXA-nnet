#pragma once

#include "utils/algorithm.hpp"

#include <mpi.h>

#include <vector>
#include <numeric>
#include <algorithm>

namespace sphexa::mpi {
	/// class correlating particles to detached data
	/**
	 * TODO
	 */
	struct mpi_partition {
		// references to pointers
		const std::vector<int> &node_id;
		const std::vector<std::size_t> &particle_id;

		// send partition limits
		std::vector<int> send_disp;
		std::vector<int> send_count;

		// send partition
		std::vector<std::size_t> send_partition;

		// send partition limits
		std::vector<int> recv_disp;
		std::vector<int> recv_count;

		// send partition
		std::vector<std::size_t> recv_partition;

		mpi_partition(const std::vector<int> &node_id_, const std::vector<std::size_t> &particle_id_) : node_id(node_id_), particle_id(particle_id_) {}

		void resize_comm_size(const int size) {
			send_disp .resize(size + 1, 0);
			send_count.resize(size    , 0);

			recv_disp. resize(size + 1, 0);
			recv_count.resize(size    , 0);
		}

		void resize_num_send(const int N) {
			send_partition.resize(N);
			recv_partition.resize(N);
		}
	};


	/// function that create a mpi partition from particle detached data pointer
	/**
	 * TODO
	 */
	mpi_partition partitionFromPointers(const std::vector<int> &node_id, const std::vector<std::size_t> &particle_id) {
		int rank, size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		mpi_partition partition(node_id, particle_id);

		// prepare vector sizes
		const int n_particles = node_id.size();
		partition.resize_comm_size(size);
		partition.send_partition.resize(n_particles);

		// localy partition
		utils::parallel_generalized_partition_from_iota(partition.send_partition.begin(), partition.send_partition.end(), 0, 
			partition.send_disp.begin(), partition.send_disp.end(),
			[&](const int idx) {
				return node_id[idx];
			});

		// send counts
		partition.recv_disp[0] = 0; partition.send_disp[0] = 0;
		std::adjacent_difference(partition.send_disp.begin() + 1, partition.send_disp.end(), partition.send_count.begin());
		MPI_Alltoall(&partition.send_count[0], 1, MPI_INT, &partition.recv_count[0], 1, MPI_INT, MPI_COMM_WORLD);
		std::partial_sum(partition.recv_count.begin(), partition.recv_count.end(), partition.recv_disp.begin() + 1);

		// send particle id
		// prepare send buffer
		std::vector<size_t> send_buffer(n_particles);
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			send_buffer[i] = particle_id[partition.send_partition[i]];

		// prepare recv buffer
		size_t n_particles_recv = partition.recv_disp[size];
		partition.recv_partition.resize(n_particles_recv);

		// send particle id
		MPI_Alltoallv(&send_buffer[0],              &partition.send_count[0], &partition.send_disp[0], MPI_UNSIGNED_LONG_LONG,
					  &partition.recv_partition[0], &partition.recv_count[0], &partition.recv_disp[0], MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		return partition;
	}

	/// initialize pointers:
	/**
	 * TODO
	 */
	void initializePointers(std::vector<int> &node_id, std::vector<std::size_t> &particle_id, const size_t local_n_particles) {
		int rank, size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		// share the number of particle per node
		std::vector<size_t> n_particles(size);
		MPI_Allgather(&local_n_particles, 1, MPI_UNSIGNED_LONG_LONG,
					  &n_particles[0],    1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

		// initialize "node_id" and "particle_id"
		size_t global_idx_begin = std::accumulate(n_particles.begin(), n_particles.begin() + rank, (size_t)0);

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i) {
			size_t global_idx = global_idx_begin + i;
			node_id[i] = global_idx % size;
			particle_id[i] = global_idx / size;
		}
	}

	/// function that sync data to detached data
	/**
	 * TODO
	 */
	template<typename T>
	void directSyncDataFromPartition(const mpi_partition &partition, const std::vector<T> &send_vector, std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		// prepare send buffer
		const int n_particles = partition.node_id.size();

		// prepare (partition) buffer
		size_t n_particles_recv = partition.recv_partition.size();
		std::vector<T> send_buffer(n_particles), recv_buffer(n_particles_recv);

		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			send_buffer[i] = send_vector[partition.send_partition[i]];

		// send buffer
		MPI_Alltoallv(&send_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype,
					  &recv_buffer[0], &partition.recv_count[0], &partition.recv_disp[0], datatype, MPI_COMM_WORLD);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_recv; ++i)
			recv_vector[partition.recv_partition[i]] = recv_buffer[i];
	}

	/// function that sync data from detached data
	/**
	 * TODO
	 */
	template<typename T>
	void reversedSyncDataFromPartition(const mpi_partition &partition, const std::vector<T> &send_vector, std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		// exact same thing as "direct_sync_data_from_partition" but with "send_ <-> recv_"

		// prepare send buffer
		const int n_particles = partition.node_id.size();

		// prepare (partition) buffer
		size_t n_particles_recv = partition.recv_partition.size();
		std::vector<T> send_buffer(n_particles_recv), recv_buffer(n_particles);

		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_recv; ++i)
			send_buffer[i] = send_vector[partition.recv_partition[i]];

		// send buffer
		MPI_Alltoallv(&send_buffer[0], &partition.recv_count[0], &partition.recv_disp[0], datatype,
					  &recv_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype, MPI_COMM_WORLD);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			recv_vector[partition.send_partition[i]] = recv_buffer[i];
	}
}