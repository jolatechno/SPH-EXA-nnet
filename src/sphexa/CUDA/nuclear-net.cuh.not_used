#pragma once

#include "../../nuclear-net.hpp"

namespace sphexa {
namespace sphnnet {
	namespace constants {
		/// number of consecutive iteration per cuda thread
		const int cuda_num_iteration_per_thread = 16;
		/// number of thread per cuda thread block
		const int cuda_num_thread_per_block = 128;
	}

	template<class func_type, class func_eos, typename Float>
	__host__ void cudaComputeNuclearReactions(const int n_particles, const int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos);
}
}