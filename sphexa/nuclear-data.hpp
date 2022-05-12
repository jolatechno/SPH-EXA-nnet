#pragma once

#include "mpi-wrapper.hpp"

#include <vector>
#include <array>

namespace sphexa::sphnnet {
	template<int n_species, typename Float>
	struct NuclearData {
		/// number of particules
		size_t n_particules = 0;

		// data to exchange data with hydro particules
		std::vector<int> node_id;
		std::vector<std::size_t> particule_id;

		/// hydro data
		std::vector<Float> rho, drho_dt, T;

		/// nuclear abundances (vector of vector)
		std::vector<std::array<Float, n_species>> Y;

		/// timesteps
		std::vector<Float> dt;

		/// resize the number of particules
		void resize(const size_t N) {
			n_particules = N;

			node_id.resize(n_particules);
			particule_id.resize(n_particules);

			rho.resize(n_particules);
			drho_dt.resize(n_particules);
			T.resize(n_particules);

			Y.resize(n_particules);

			dt.resize(n_particules, 1e-12);
		}



		/// update node id and particule id of particule data
		template<class ParticuleData>
		void update_particule_pointers(ParticuleData &d) const {
			sync_pointers(node_id, particule_id, d.node_id, d.particule_id);
		}

		/// update node id and particule id of nuclear data
		template<class ParticuleData>
		void update_nuclear_pointers(const ParticuleData &d) {
			sync_pointers(d.node_id, d.particule_id, node_id, particule_id);
		}



		/// send hydro data
		template<class ParticuleData>
		void update_nuclear_data(const ParticuleData &d) {
			sync_data_from_pointers(d.node_id, d.particule_id, d.rho,     rho);
			sync_data_from_pointers(d.node_id, d.particule_id, d.drho_dt, drho_dt);
			sync_data_from_pointers(d.node_id, d.particule_id, d.T,       T);
		}

		/// send back hydro data
		template<class ParticuleData>
		void update_hydro_data(ParticuleData &d) const {
			sync_data_from_pointers(node_id, particule_id, rho,     d.rho);
			sync_data_from_pointers(node_id, particule_id, drho_dt, d.drho_dt);
			sync_data_from_pointers(node_id, particule_id, T,       d.T);
		}
	};
}