#pragma once

namespace nnet {
	// forward definition
	class ptr_reaction_list;
}

namespace sphexa {
namespace sphnnet {
	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(int n_particles, int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos);
}
}