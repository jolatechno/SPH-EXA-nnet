#pragma once

#include "nuclear-data.hpp"
#include "../src/nuclear-net.hpp"

namespace sphexa {
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