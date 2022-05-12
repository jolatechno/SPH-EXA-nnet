#pragma once

#include <mpi.h>

#include <vector>
#include <numeric>

namespace sphexa {
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
	};


	mpi_partition partition_from_pointers(const std::vector<int> &node_id, const std::vector<std::size_t> &particule_id) {
		mpi_partition partition;

		// intialize references
		partition.node_id = &node_id;
		partition.particule_id = &particule_id;

		/* TODO */

		return partition;
	}

	template<typename T>
	void direct_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, const std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		/* TODO */
	}

	template<typename T>
	void reversed_sync_data_from_partition(const mpi_partition &partition, const std::vector<T> &send_vector, const std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		/* TODO */
	}
}