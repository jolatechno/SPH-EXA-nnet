#pragma once

#include <mpi.h>

#include <vector>
#include <numeric>

namespace sphexa {
	template<typename T>
	void sync_data_from_pointers(const std::vector<int> &node_id, const std::vector<size_t> &particule_id, const std::vector<T> &send_vector, const std::vector<T> &recv_vector, const MPI_Datatype datatype) {
		/* TODO */
	}

	void sync_pointers(const std::vector<int> &node_id, const std::vector<size_t> &particule_id, std::vector<int> &to_update_node_id, std::vector<size_t> &to_update_particule_id) {
		// get this node id
		int this_node_id;
		MPI_Comm_rank(MPI_COMM_WORLD, &this_node_id);

		// initialize node id
		std::vector<int> send_node_id(node_id.size());
		std::fill(send_node_id.begin(), send_node_id.end(), this_node_id);

		// initialize particule id
		std::vector<size_t> send_particule_id(particule_id.size());
		std::iota(send_particule_id.begin(), send_particule_id.end(), 0);

		// send node id
		sync_data_from_pointers(node_id, particule_id, send_node_id, to_update_node_id, MPI_INT);

		// send particule id
		sync_data_from_pointers(node_id, particule_id, send_particule_id, to_update_particule_id, MPI_UNSIGNED_LONG_LONG);
	}
}