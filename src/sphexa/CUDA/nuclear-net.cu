 #include "device_launch_parameters.h"

namespace sphexa {
namespace sphnnet {
	template<class func_type, class func_eos, typename Float>
	__global__ static void computeNuclearReactions(int dimension,
	Float *rho_, Float *previous_rho_, Float *Y_, Float *temp_, Float *dt_,
	const Float hydro_dt, const Float previous_dt,
		const nnet::ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos)
	{
	    int i = threadIdx.x;
	    
	    if (rho_[i] > nnet::constants::min_rho && temp_[i] > nnet::constants::min_temp) {
			// compute drho/dt
			Float drho_dt = previous_rho_[i] <= 0 ? 0. : (rho_[i] - previous_rho_[i])/previous_dt;

			// solve
			nnet::solve_system_substep(dimension,
				reactions, construct_rates_BE, eos,
				&Y_[dimension*i], temp_[i],
				rho_[i], drho_dt, hydro_dt, dt_[i]);
		}
	}
}
}