#pragma once

#include "mpi-wrapper.hpp"

#include <vector>
#include <array>

namespace sphexa::sphnnet {
	/// nuclear abundances type, that is integrated into NuclearData, or should be integrated into ParticlesData
	template <int n_species, typename Float=double>
	using NuclearAbundances = std::array<Float, n_species>;

	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double, MPI_Datatype datatype=MPI_DOUBLE>
	struct NuclearDataType {
		/// hydro data
		std::vector<Float> rho, drho_dt, T;

		/// nuclear abundances (vector of vector)
		std::vector<NuclearAbundances<n_species, Float>> Y;

		/// timesteps
		std::vector<Float> dt;

		/// resize the number of particules
		void resize(const size_t N) {
			rho.resize(N);
			drho_dt.resize(N);
			T.resize(N);

			Y.resize(N);

			dt.resize(N, 1e-12);
		}



		/// receive hydro data
		template<class ParticlesDataType>
		void getParticulesValue(ParticlesDataType &d, const mpi_partition &partition) const {
			direct_sync_data_from_partition(partition, d.rho,     rho,     datatype);
			direct_sync_data_from_partition(partition, d.drho_dt, drho_dt, datatype);
			direct_sync_data_from_partition(partition, d.T,       T,       datatype);
		}

		/// send back hydro data
		template<class ParticlesDataType>
		void sendParticulesValue(ParticlesDataType &d, const mpi_partition &partition) const {
			inverse_sync_data_from_partition(partition, rho,     d.rho,     datatype);
			inverse_sync_data_from_partition(partition, drho_dt, d.drho_dt, datatype);
			inverse_sync_data_from_partition(partition, T,       d.T,       datatype);
		}
	};
}