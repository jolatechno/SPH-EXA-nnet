#include "nnet/net87/net87.cuh"
#include "nnet/net86/net86.cuh"
#include "nnet/net14/net14.cuh"

#include "nnet/eos/helmholtz.cuh"
#include "nnet/eos/ideal_gas.cuh"

namespace nnet {
	namespace net87 {
		namespace electrons::constants {
			__device__ double dev_log_temp_ref[N_TEMP];
			__device__ double dev_log_rho_ref[N_RHO];
			__device__ double dev_electron_rate[N_TEMP][N_RHO][N_C];
		}
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace net86 {
		bool debug = false;
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace net14 {
		bool debug = false;
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace eos {
		bool debug = false;
	}
}