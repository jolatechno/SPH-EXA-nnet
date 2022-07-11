#include "electrons.hpp"

namespace nnet::net87::electrons {
	namespace constants {
		__device__ double dev_log_temp_ref[N_TEMP];
		__device__ double dev_log_rho_ref[N_RHO];
		__device__ double dev_electron_rate[N_TEMP][N_RHO][N_C];
	}
}