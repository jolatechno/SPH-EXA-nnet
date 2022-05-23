#pragma once

#include <numeric>

#include "nuclear-data.hpp"

#ifdef USE_MPI
	#include "mpi/mpi-wrapper.hpp"
#endif

#include "../nuclear-net.hpp"

namespace sphexa::sphnnet {
	/// function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y.
	/**
	 * TODO
	 */
	template<class Data, class func_rate, class func_BE, class func_eos, typename Float>
	void compute_nuclear_reactions(Data &n, const Float hydro_dt, const Float previous_dt,
		const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {
		const size_t n_particles = n.temp.size();

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n_particles; ++i) {
			Float drho_dt = (n.rho[i] - n.previous_rho[i])/previous_dt;

			std::tie(n.Y[i], n.temp[i]) = nnet::solve_system_substep(reactions, construct_rates, construct_BE, eos,
				n.Y[i], n.temp[i],
				n.rho[i], drho_dt, hydro_dt, n.dt[i]);
		}
	}

#ifdef USE_MPI
	/// function initializing the partition of NuclearDataType from 
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void initializePartition(const ParticlesDataType &d, NuclearDataType<n_species, Float> &n) {
		n.partition = sphexa::mpi::partitionFromPointers(d.node_id, d.particle_id, d.comm);
	}


	/// function sending requiered hydro data from ParticlesDataType to NuclearDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void sendHydroData(const ParticlesDataType &d, NuclearDataType<n_species, Float> &n) {
		if (!n.first_step)
			std::swap(n.rho, n.previous_rho);

		sphexa::mpi::directSyncDataFromPartition(n.partition, d.rho,  n.rho,  d.comm);
		sphexa::mpi::directSyncDataFromPartition(n.partition, d.temp, n.temp, d.comm);

		if (n.first_step) {
			std::copy(n.rho.begin(), n.rho.end(), n.previous_rho.begin());
			n.first_step = false;
		}	
	}

	/// sending back hydro data from NuclearDataType to ParticlesDataType
	/**
	 * TODO
	 */
	template<class ParticlesDataType, int n_species, typename Float=double>
	void recvHydroData(ParticlesDataType &d, const NuclearDataType<n_species, Float> &n) {
		sphexa::mpi::reversedSyncDataFromPartition(n.partition, n.temp, d.temp, d.comm);
	}
#endif
}