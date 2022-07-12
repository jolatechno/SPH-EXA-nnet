#include "nnet/net87/net87.cuh"
#include "nnet/net86/net86.cuh"
#include "nnet/net14/net14.cuh"

#include "nnet/eos/helmholtz.cuh"
#include "nnet/eos/ideal_gas.cuh"

namespace nnet {
	namespace net87 {
		namespace electrons::constants {
			DEVICE_FINALIZE_EXTERN(double, log_temp_ref[N_TEMP], ;)
        	DEVICE_FINALIZE_EXTERN(double, log_rho_ref[N_RHO], ;)
        	DEVICE_FINALIZE_EXTERN(double, electron_rate[N_TEMP][N_RHO][N_C], ;)
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
		namespace helmholtz_constants {
			DEVICE_FINALIZE_EXTERN(double, d[IMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, dd_sav[IMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dd2_sav[IMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, ddi_sav[IMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dd2i_sav[IMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dd3i_sav[IMAX - 1], ;)

	        DEVICE_FINALIZE_EXTERN(double, t[JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, dt_sav[JMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dt2_sav[JMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dti_sav[JMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dt2i_sav[JMAX - 1], ;)
	        DEVICE_FINALIZE_EXTERN(double, dt3i_sav[JMAX - 1], ;)

	        DEVICE_FINALIZE_EXTERN(double, f[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fd[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, ft[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fdd[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, ftt[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fdt[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fddt[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fdtt[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, fddtt[IMAX][JMAX], ;)

	        DEVICE_FINALIZE_EXTERN(double, dpdf[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, dpdfd[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, dpdft[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, dpdfdt[IMAX][JMAX], ;)

	        DEVICE_FINALIZE_EXTERN(double, ef[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, efd[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, eft[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, efdt[IMAX][JMAX], ;)

	        DEVICE_FINALIZE_EXTERN(double, xf[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, xfd[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, xft[IMAX][JMAX], ;)
	        DEVICE_FINALIZE_EXTERN(double, xfdt[IMAX][JMAX], ;)
		}
		bool debug = false;
	}
}