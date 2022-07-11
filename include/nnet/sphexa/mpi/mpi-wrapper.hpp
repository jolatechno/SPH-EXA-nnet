#pragma once

#include "../util/algorithm.hpp"

#ifdef USE_MPI
	#include <mpi.h>
#endif

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
#ifdef USE_MPI
		int rank, size;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);

		mpi_partition partition(node_id, particle_id);

		// prepare vector sizes
		const int n_particles = lastIndex - firstIndex;
		partition.resize_comm_size(size);
		partition.send_partition.resize(n_particles);

		// localy partition
		util::parallel_generalized_partition_from_iota(partition.send_partition.begin(), partition.send_partition.end(), firstIndex, 
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
		size_t n_particles_recv = partition.recv_disp[size];
		partition.recv_partition.resize(n_particles_recv);

		// send particle id
		MPI_Alltoallv(&send_buffer[0],              &partition.send_count[0], &partition.send_disp[0], MPI_UNSIGNED_LONG_LONG,
					  &partition.recv_partition[0], &partition.recv_count[0], &partition.recv_disp[0], MPI_UNSIGNED_LONG_LONG, comm);

		return partition;
#endif
	}



#ifdef USE_MPI
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
#endif


#ifdef USE_MPI
	/// function that sync data to detached data
	/**
	 * TODO
	 */
	template<typename T>
	void syncDataToStaticPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, const MPI_Datatype datatype, MPI_Comm comm) {
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
	void syncDataToStaticPartition(const mpi_partition &partition, const double *send_vector, double *recv_vector, MPI_Comm comm) {
		syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
	}
	void syncDataToStaticPartition(const mpi_partition &partition, const float *send_vector, float *recv_vector, MPI_Comm comm) {
		syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
	}
	void syncDataToStaticPartition(const mpi_partition &partition, const int *send_vector, int *recv_vector, MPI_Comm comm) {
		syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_INT, comm);
	}
	void syncDataToStaticPartition(const mpi_partition &partition, const uint *send_vector, uint *recv_vector, MPI_Comm comm) {
		syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
	}
	void syncDataToStaticPartition(const mpi_partition &partition, const size_t *send_vector, size_t *recv_vector, MPI_Comm comm) {
		syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
	}
	template<typename T>
	void syncDataToStaticPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type not implictly supported by syncDataToStaticPartition\n");
	}
	template<typename T1, typename T2>
	void syncDataToStaticPartition(const mpi_partition &partition, const T1 *send_vector, T2 *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type mismatch in syncDataToStaticPartition\n");
	}

	/// function that sync data from detached data
	/**
	 * TODO
	 */
	template<typename T>
	void syncDataFromStaticPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, const MPI_Datatype datatype, MPI_Comm comm) {
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
	void syncDataFromStaticPartition(const mpi_partition &partition, const double *send_vector, double *recv_vector, MPI_Comm comm)  {
		syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
	}
	void syncDataFromStaticPartition(const mpi_partition &partition, const float *send_vector, float *recv_vector, MPI_Comm comm)  {
		syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
	}
	void syncDataFromStaticPartition(const mpi_partition &partition, const int *send_vector, int *recv_vector, MPI_Comm comm)  {
		syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_INT, comm);
	}
	void syncDataFromStaticPartition(const mpi_partition &partition, const uint *send_vector, uint *recv_vector, MPI_Comm comm)  {
		syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
	}
	void syncDataFromStaticPartition(const mpi_partition &partition, const size_t *send_vector, size_t *recv_vector, MPI_Comm comm)  {
		syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
	}
	template<typename T>
	void syncDataFromStaticPartition(const mpi_partition &partition, const T *send_vector, T *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type not implictly supported by syncDataFromStaticPartition\n");
	}
	template<typename T1, typename T2>
	void syncDataFromStaticPartition(const mpi_partition &partition, const T1 *send_vector, T2 *recv_vector, MPI_Comm comm) {
		throw std::runtime_error("Type mismatch in syncDataFromStaticPartition\n");
	}
#endif



	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void syncDataToStaticPartition(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		// get data
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			// send
			std::visit(
				[&d, &n](auto&& send, auto &&recv){
#ifdef USE_MPI
					syncDataToStaticPartition(n.partition, send->data(), recv->data(), d.comm);
#else
					if constexpr (std::is_same<decltype(send), decltype(recv)>::value)
						*recv = *send;
#endif
				}, particleData[particleFieldIdx], nuclearData[nuclearFieldIdx]);
		}
	}



	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, class nuclearDataType>
	void syncDataFromStaticPartition(ParticlesDataType &d, nuclearDataType &n, const std::vector<std::string> &sync_fields) {
		auto nuclearData  = n.data();
		auto particleData = d.data();

		// send fields
		for (auto field : sync_fields) {
			// find field
			int nuclearFieldIdx = std::distance(n.fieldNames.begin(), 
				std::find(n.fieldNames.begin(), n.fieldNames.end(), field));
			int particleFieldIdx = std::distance(d.fieldNames.begin(), 
				std::find(d.fieldNames.begin(), d.fieldNames.end(), field));

			std::visit(
				[&d, &n](auto&& send, auto &&recv){
#ifdef USE_MPI
					syncDataFromStaticPartition(n.partition, send->data(), recv->data(), d.comm);
#else
					if constexpr (std::is_same<decltype(send), decltype(recv)>::value)
						*recv = *send;
#endif
				}, nuclearData[nuclearFieldIdx], particleData[particleFieldIdx]);
		}
	}
}