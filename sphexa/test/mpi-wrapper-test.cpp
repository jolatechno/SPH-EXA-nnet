#include <vector>

#include "../src/mpi-wrapper.hpp"

int main(int argc, char* argv[]) {
	int size, rank;
    MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<float> x, x_out(4);
	std::vector<int> node_id;
	std::vector<size_t> particle_id;

	if (rank == 0) {
		node_id     = {0, 1, 1, 0};
		particle_id = {2, 2, 0, 1};

		x = {0.0, 0.1, 0.2, 0.3};
	} else if (rank == 1) {
		node_id     = {0, 1, 0, 1};
		particle_id = {0, 3, 3, 1};

		x = {1.0, 1.1, 1.2, 1.3};
	}

	auto partition = sphexa::mpi::partitionFromPointers(node_id, particle_id);

	sphexa::mpi::directSyncDataFromPartition(partition, x, x_out, MPI_FLOAT);

	if (rank < 2) {
		std::cout << rank << "\tdirect  \t";
		for (int i = 0; i < 4; ++i)
			std::cout << x_out[i] << ", ";
		std::cout << "\n";
	}

	sphexa::mpi::reversed_sync_data_from_partition(partition, x_out, x, MPI_FLOAT);

	if (rank < 2) {
		std::cout << rank << "\treversed\t";
		for (int i = 0; i < 4; ++i)
			std::cout << x[i] << ", ";
		std::cout << "\n";
	}

	MPI_Finalize();

	return 0;
}