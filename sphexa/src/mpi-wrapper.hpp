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
		std::vector<int> send_begin;
		std::vector<int> send_count;

		// send partition
		std::vector<std::size_t> send_partition;

		// send partition limits
		std::vector<int> recv_begin;
		std::vector<int> recv_count;

		// send partition
		std::vector<std::size_t> recv_partition;

		void resize_comm_size(const int size) {
			send_begin.resize(size + 1, 0);
			send_count.resize(size    , 0);

			recv_begin.resize(size + 1, 0);
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
		partition.resize_comm_size(size);
		partition.resize_num_particles(node_id.size());

		// intialize references
		partition.node_id = &node_id;
		partition.particule_id = &particule_id;

		// localy partition
		/* TODO */

		// send counts
		std::adjacent_difference(partition.send_begin.begin() + 1, partition.send_begin.end(), partition.send_count.begin());
		/* TODO */
		std::partial_sum(partition.recv_count.begin(), partition.recv_count.end(), partition.recv_begin.begin() + 1);

		// send particle id
		/* TODO */

		return partition;
	}

	/// function that sync data to detached data
	/**
	 * TODO
	 */
	template<typename T>
	void direct_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, const std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		/* TODO */
	}

	/// function that sync data from detached data
	/**
	 * TODO
	 */
	template<typename T>
	void reversed_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, const std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		/* TODO */
	}
}