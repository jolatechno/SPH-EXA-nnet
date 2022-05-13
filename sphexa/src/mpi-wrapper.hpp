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
		const std::vector<int> *node_id;
		const std::vector<std::size_t> *particule_id;

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

		void resize_comm_size(const int size) {
			send_disp.resize(size + 1, 0);
			send_count.resize(size    , 0);

			recv_disp.resize(size + 1, 0);
			recv_count.resize(size    , 0);
		}

		void resize_num_particles(const int N) {
			send_partition.resize(N);
			recv_partition.resize(N);
		}
	};


	/// function that create a mpi partition from particle detached data pointer
	/**
	 * TODO
	 */
	mpi_partition partition_from_pointers(const std::vector<int> &node_id, const std::vector<std::size_t> &particule_id) {
		int rank, size;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		mpi_partition partition;

		// prepare vector sizes
		const int n_particles = node_id.size();
		partition.resize_comm_size(size);
		partition.resize_num_particles(n_particles);

		// intialize references
		partition.node_id = &node_id;
		partition.particule_id = &particule_id;

		// localy partition
		utils::parallel_generalized_partition_from_iota(partition.send_partition.begin(), partition.send_partition.end(), 0, 
			partition.send_disp.begin(), partition.send_disp.end(),
			[&](const int idx) {
				return node_id[idx];
			});

		// send counts
		std::adjacent_difference(partition.send_disp.begin() + 1, partition.send_disp.end(), partition.send_count.begin());
		MPI_Alltoall(&partition.send_count[0], 1, MPI_INT, &partition.recv_count[0], 1, MPI_INT, MPI_COMM_WORLD);
		std::partial_sum(partition.recv_count.begin(), partition.recv_count.end(), partition.recv_disp.begin() + 1);

		// send particle id
		// prepare send buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			partition.recv_partition[i] = particule_id[partition.send_partition[i]];
		// send particle id
		MPI_Alltoallv(&partition.recv_partition[0], &partition.send_count[0], &partition.send_disp[0], MPI_INT,
					   MPI_IN_PLACE,                &partition.recv_count[0], &partition.recv_disp[0], MPI_INT, MPI_COMM_WORLD);

		return partition;
	}

	/// function that sync data to detached data
	/**
	 * TODO
	 */
	template<typename T>
	void direct_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		// prepare send buffer
		const int n_particles = partition.node_id->size();
		std::vector<T> send_recv_buffer(n_particles);

		// prepare (partition) buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			send_recv_buffer[i] = send_vector[partition.send_partition[i]];

		// send buffer
		MPI_Alltoallv(&send_recv_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype,
					   MPI_IN_PLACE,        &partition.recv_count[0], &partition.recv_disp[0], datatype, MPI_COMM_WORLD);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			recv_vector[i] = send_vector[partition.recv_partition[i]];
	}

	/// function that sync data from detached data
	/**
	 * TODO
	 */
	template<typename T>
	void reversed_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		// exact same thing as "direct_sync_data_from_partition" but with "send_ <-> recv_"

		// prepare send buffer
		const int n_particles = partition.node_id->size();
		std::vector<T> send_recv_buffer(n_particles);

		// prepare (partition) buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			send_recv_buffer[partition.recv_partition[i]] = send_vector[i];

		// send buffer
		MPI_Alltoallv(&send_recv_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype,
					   MPI_IN_PLACE,        &partition.recv_count[0], &partition.recv_disp[0], datatype, MPI_COMM_WORLD);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles; ++i)
			recv_vector[partition.recv_partition[i]] = send_vector[i];
	}
}