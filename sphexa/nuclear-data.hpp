#pragma once

#include "mpi-wrapper.hpp"

#include <vector>
#include <array>

namespace sphexa::sphnnet {
	/// nuclear abundances type, that is integrated into NuclearData, or should be integrated into ParticlesData
	template <int n_species, typename Float>
	using NuclearAbundances = std::array<Float, n_species>;

	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<int n_species, typename Float>
	struct NuclearData {
		// data to exchange data with hydro particules
		std::vector<int> node_id;
		std::vector<std::size_t> particule_id;

		/// hydro data
		std::vector<Float> rho, drho_dt, T;

		/// nuclear abundances (vector of vector)
		std::vector<NuclearAbundances<n_species, Float>> Y;

		/// timesteps
		std::vector<Float> dt;

		/// resize the number of particules
		void resize(const size_t N) {
			node_id.resize(N);
			particule_id.resize(N);

			rho.resize(N);
			drho_dt.resize(N);
			T.resize(N);

			Y.resize(N);

			dt.resize(N, 1e-12);
		}



		/// update node id and particule id of particule data
		template<class ParticlesData>
		void update_particule_pointers(ParticlesData &d) const {
			sync_pointers(node_id, particule_id, d.node_id, d.particule_id);
		}

		/// update node id and particule id of nuclear data
		template<class ParticlesData>
		void update_nuclear_pointers(const ParticlesData &d) {
			sync_pointers(d.node_id, d.particule_id, node_id, particule_id);
		}



		/// send hydro data
		template<class ParticlesData>
		void update_nuclear_data(const ParticlesData &d) {
			sync_data_from_pointers(d.node_id, d.particule_id, d.rho,     rho);
			sync_data_from_pointers(d.node_id, d.particule_id, d.drho_dt, drho_dt);
			sync_data_from_pointers(d.node_id, d.particule_id, d.T,       T);
		}

		/// send back hydro data
		template<class ParticlesData>
		void update_hydro_data(ParticlesData &d) const {
			sync_data_from_pointers(node_id, particule_id, rho,     d.rho);
			sync_data_from_pointers(node_id, particule_id, drho_dt, d.drho_dt);
			sync_data_from_pointers(node_id, particule_id, T,       d.T);
		}
	};
}