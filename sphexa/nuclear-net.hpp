#pragma once

#include "nuclear-data.hpp"
#include "../src/nuclear-net.hpp"

namespace sphexa::sphnnet {
	/// function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y.
	/**
	 * TODO
	 */
	template<class Data, class Vector, class func_rate, class func_BE, class func_eos, typename Float>
	void compute_nuclear_reactions(Data &n, const Float hydro_dt,
		const std::vector<nnet::reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos) {

		#pragma omp parallel for schedule(dynamic)
		for (size_t i = 0; i < n.Y.size(); ++i)
			n.Y[i], n.T[i] = nnet::solve_system_superstep(reactions, construct_rates, construct_BE, eos,
				n.Y[i], n.T[i], n.rho[i], n.drho_dt[i], hydro_dt, n.dt[i]);
	}
}

/* modifications to the "ParticlesData" class:

#ifdef ATTACH_NUCLEAR_DATA
	template<int n_species, typename Float>
#endif
class ParticlesData {
	// ...

#ifdef ATTACH_NUCLEAR_DATA
	// for the "attached" implementation, i.e. nuclear data are stored on each particles:

	/// nuclear abundances (vector of vector)
	std::vector<sphexa::sphnnet::NuclearAbundances<n_species, Float>> Y;

	/// timesteps
	std::vector<Float> dt;

#else
	// for the "detached" implementation; i.e. nuclear abundances are on different nodes as the hydro data:

	// pointers to exchange data with hydro particules
	std::vector<int> node_id;
	std::vector<std::size_t> particule_id;
#endif

	// ...
}


additions to the "step" function of SPH-EXA:

void step(DomainType& domain, ParticleDataType& d
#ifndef ATTACH_NUCLEAR_DATA	
	, NuclearDataType& n
#endif
	) override {

	// ...
	// compute eos

#ifdef ATTACH_NUCLEAR_DATA
	sphexa::compute_nuclear_reactions(d, hydro_dt, ...);
#else
	n.update_particule_pointers(d);
	n.update_nuclear_data(d);
	sphexa::compute_nuclear_reactions(n, hydro_dt, ...);
	n.update_hydro_data(d);
#endif

	// ...
}
*/