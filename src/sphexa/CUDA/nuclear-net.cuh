#pragma once

using Float      = double;
using eos_struct = struct eos_output {
			Float cv, dP_dT, p;
			Float cp, c, u;
	};
using func_eos   = eos_struct (*) (const Float *, const Float, const Float);
using func_type  = void     (*) (const Float *, const Float, const Float, eos_struct, const Float *, const Float *, const Float *);

extern "C" {
namespace nnet {
	// forward definition
	class ptr_reaction_list;
}
}

namespace sphexa {
namespace sphnnet {
	// template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const int n_particles, const int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos);
}
}