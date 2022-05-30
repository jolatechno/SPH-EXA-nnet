#pragma once

#include "../util/algorithm.hpp"

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
		const std::vector<int>             *node_id;
		const std::vector<std::size_t>     *particle_id;

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

		mpi_partition() {}
		mpi_partition(const std::vector<int> &node_id_, const std::vector<std::size_t> &particle_id_) : node_id(&node_id_), particle_id(&particle_id_) {}

		void resize_comm_size(const int size) {
			send_disp .resize(size + 1, 0);
			send_count.resize(size,     0);

			recv_disp. resize(size + 1, 0);
			recv_count.resize(size,     0);
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
	mpi_partition partitionFromPointers(size_t firstIndex, size_t lastIndex, const std::vector<int> &node_id, const std::vector<std::size_t> &particle_id, MPI_Comm comm)
	{
		int rank, size;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);

		mpi_partition partition(node_id, particle_id);

		// prepare vector sizes
		const int n_particles = lastIndex - firstIndex;
		partition.resize_comm_size(size);
		partition.send_partition.resize(n_particles);

		// localy partition
		utils::parallel_generalized_partition_from_iota(partition.send_partition.begin(), partition.send_partition.end(), firstIndex, 
			partition.send_disp.begin(), partition.send_disp.end(),
			[&](const int idx) {
				return node_id[idx];
			});

		// send counts
		partition.recv_disp[0] = 0; partition.send_disp[0] = 0;
		std::adjacent_difference(partition.send_disp.begin() + 1, partition.send_disp.end(), partition.send_count.begin());
		MPI_Alltoall(&partition.send_count[0], 1, MPI_INT, &partition.recv_count[0], 1, MPI_INT, comm);
		std::partial_sum(partition.recv_count.begin(), partition.recv_count.end(), partition.recv_disp.begin() + 1);

		// send particle id
		// prepare send buffer
		size_t n_particles_send = partition.send_disp[size];
		std::vector<size_t> send_buffer(n_particles_send);
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_send; ++i)
			send_buffer[i] = particle_id[partition.send_partition[i]];

		// prepare recv buffer
		size_t n_particles_recv  = partition.recv_disp[size];
		partition.recv_partition.resize(n_particles_recv);

		// send particle id
		MPI_Alltoallv(&send_buffer[0],              &partition.send_count[0], &partition.send_disp[0], MPI_UNSIGNED_LONG_LONG,
					  &partition.recv_partition[0], &partition.recv_count[0], &partition.recv_disp[0], MPI_UNSIGNED_LONG_LONG, comm);

		return partition;
	}



	/// initialize pointers:
	/**
	 * TODO
	 */
	void initializePointers(size_t firstIndex, size_t lastIndex, std::vector<int> &node_id, std::vector<std::size_t> &particle_id, MPI_Comm comm) {
		int rank, size;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);

		// share the number of particle per node
		std::vector<size_t> n_particles(size);
		size_t local_n_particles = lastIndex - firstIndex;
		MPI_Allgather(&local_n_particles, 1, MPI_UNSIGNED_LONG_LONG,
					  &n_particles[0],    1, MPI_UNSIGNED_LONG_LONG, comm);

		// initialize "node_id" and "particle_id"
		size_t global_idx_begin = std::accumulate(n_particles.begin(), n_particles.begin() + rank, (size_t)0);

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < local_n_particles; ++i) {
			size_t global_idx = global_idx_begin + i;
			node_id[    firstIndex + i] = global_idx % size;
			particle_id[firstIndex + i] = global_idx / size;
		}
	}

	/// function that sync data to detached data
	/**
	 * TODO
	 */
	template<typename T>
	void directSyncDataFromPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, const MPI_Datatype datatype, MPI_Comm comm) {
		int size;
		MPI_Comm_size(comm, &size);

		const size_t n_particles_send = partition.send_disp[size];
		const size_t n_particles_recv = partition.recv_disp[size];

		// prepare (partition) buffer
		std::vector<T> send_buffer(n_particles_send), recv_buffer(n_particles_recv);

		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_send; ++i)
			send_buffer[i] = send_vector[partition.send_partition[i]];

		// send buffer
		MPI_Alltoallv(&send_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype,
					  &recv_buffer[0], &partition.recv_count[0], &partition.recv_disp[0], datatype, comm);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_recv; ++i)
			recv_vector[partition.recv_partition[i]] = recv_buffer[i];
	}
	void directSyncDataFromPartition(const mpi_partition &partition, const double *send_vector, double *recv_vector, MPI_Comm comm) {
		directSyncDataFromPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
	}
	void directSyncDataFromPartition(const mpi_partition &partition, const float *send_vector, float *recv_vector, MPI_Comm comm) {
		directSyncDataFromPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
	}
	void directSyncDataFromPartition(const mpi_partition &partition, const int *send_vector, int *recv_vector, MPI_Comm comm) {
		directSyncDataFromPartition(partition, send_vector, recv_vector, MPI_INT, comm);
	}
	void directSyncDataFromPartition(const mpi_partition &partition, const uint *send_vector, uint *recv_vector, MPI_Comm comm) {
		directSyncDataFromPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
	}
	void directSyncDataFromPartition(const mpi_partition &partition, const size_t *send_vector, size_t *recv_vector, MPI_Comm comm) {
		directSyncDataFromPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
	}
	template<typename T>
	void directSyncDataFromPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type not implictly supported by directSyncDataFromPartition\n");
	}
	template<typename T1, typename T2>
	void directSyncDataFromPartition(const mpi_partition &partition, const T1 *send_vector, T2 *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type mismatch in directSyncDataFromPartition\n");
	}

	/// function that sync data from detached data
	/**
	 * TODO
	 */
	template<typename T>
	void reversedSyncDataFromPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, const MPI_Datatype datatype, MPI_Comm comm) {
		// exact same thing as "direct_sync_data_from_partition" but with "send_ <-> recv_"

		int size;
		MPI_Comm_size(comm, &size);

		const size_t n_particles_send = partition.send_disp[size];
		const size_t n_particles_recv = partition.recv_disp[size];

		// prepare (partition) buffer
		std::vector<T> send_buffer(n_particles_recv), recv_buffer(n_particles_send);

		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_recv; ++i)
			send_buffer[i] = send_vector[partition.recv_partition[i]];

		// send buffer
		MPI_Alltoallv(&send_buffer[0], &partition.recv_count[0], &partition.recv_disp[0], datatype,
					  &recv_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype, comm);

		// reconstruct (un-partition) vector from buffer
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n_particles_send; ++i)
			recv_vector[partition.send_partition[i]] = recv_buffer[i];
	}
	void reversedSyncDataFromPartition(const mpi_partition &partition, const double *send_vector, double *recv_vector, MPI_Comm comm)  {
		reversedSyncDataFromPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
	}
	void reversedSyncDataFromPartition(const mpi_partition &partition, const float *send_vector, float *recv_vector, MPI_Comm comm)  {
		reversedSyncDataFromPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
	}
	void reversedSyncDataFromPartition(const mpi_partition &partition, const int *send_vector, int *recv_vector, MPI_Comm comm)  {
		reversedSyncDataFromPartition(partition, send_vector, recv_vector, MPI_INT, comm);
	}
	void reversedSyncDataFromPartition(const mpi_partition &partition, const uint *send_vector, uint *recv_vector, MPI_Comm comm)  {
		reversedSyncDataFromPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
	}
	void reversedSyncDataFromPartition(const mpi_partition &partition, const size_t *send_vector, size_t *recv_vector, MPI_Comm comm)  {
		reversedSyncDataFromPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
	}
	template<typename T>
	void reversedSyncDataFromPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type not implictly supported by reversedSyncDataFromPartition\n");
	}
	template<typename T1, typename T2>
	void reversedSyncDataFromPartition(const mpi_partition &partition, const T1 *send_vector, T2 *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type mismatch in reversedSyncDataFromPartition\n");
	}
}