#pragma once

#include "mpi-wrapper.hpp"
#include "nuclear-data.hpp"

namespace sphexa::sphnnet {
	template<int n_species, typename Float=double>
	class NuclearIoDataSet {
	public:
		const std::vector<NuclearAbundances<n_species, Float>> &Y;
		std::vector<Float> x, y, z;

		NuclearIoDataSet(const NuclearDataType<n_species, Float> &n) : Y(n.Y) {}

		void resize(size_t const N) {
			x.resize(N);
			y.resize(N);
			z.resize(N);
		}
	}

	/// intialize nuclear data, from a function of positions:
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, class ParticlesDataType>
	NuclearIoDataSet<n_species, Float> initIoDataset(const NuclearDataType<n_species, Float> &n, const ParticlesDataType &d, const sphexa::mpi::mpi_partition &partition, MPI_Datatype datatype) {
		const size_t local_nuclear_n_particles = partition.recv_partition.size();

		NuclearIoDataSet dataset(n);

		// share the initial rho
		n.resize(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.rho, n.rho, datatype);

		// receiv position for initializer
		std::vector<Float> x(local_nuclear_n_particles), y(local_nuclear_n_particles), z(local_nuclear_n_particles);
		sphexa::mpi::directSyncDataFromPartition(partition, d.x, dataset.x, datatype);
		sphexa::mpi::directSyncDataFromPartition(partition, d.y, dataset.y, datatype);
		sphexa::mpi::directSyncDataFromPartition(partition, d.z, dataset.z, datatype);

		return dataset;
	}
}