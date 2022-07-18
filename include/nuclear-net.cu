#include "nnet/net87/net87.hpp"
#include "nnet/net86/net86.hpp"
#include "nnet/net14/net14.hpp"

#include "nnet/eos/helmholtz.hpp"
#include "nnet/eos/ideal_gas.hpp"

namespace nnet {
	namespace net87 {
		namespace electrons::constants {
			DEVICE_DEFINE(double, log_temp_ref[N_TEMP], ;)
	        DEVICE_DEFINE(double, log_rho_ref[N_RHO], ;)
	        DEVICE_DEFINE(double, electron_rate[N_TEMP][N_RHO][N_C], ;)
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
			DEVICE_DEFINE(double, d[IMAX], ;)
	        DEVICE_DEFINE(double, dd_sav[IMAX - 1], ;)
	        DEVICE_DEFINE(double, dd2_sav[IMAX - 1], ;)
	        DEVICE_DEFINE(double, ddi_sav[IMAX - 1], ;)
	        DEVICE_DEFINE(double, dd2i_sav[IMAX - 1], ;)
	        DEVICE_DEFINE(double, dd3i_sav[IMAX - 1], ;)

	        DEVICE_DEFINE(double, t_[JMAX], ;)
	        DEVICE_DEFINE(double, dt_sav[JMAX - 1], ;)
	        DEVICE_DEFINE(double, dt2_sav[JMAX - 1], ;)
	        DEVICE_DEFINE(double, dti_sav[JMAX - 1], ;)
	        DEVICE_DEFINE(double, dt2i_sav[JMAX - 1], ;)
	        DEVICE_DEFINE(double, dt3i_sav[JMAX - 1], ;)

	        DEVICE_DEFINE(double, f[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fd[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, ft[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fdd[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, ftt[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fdt[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fddt[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fdtt[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, fddtt[IMAX][JMAX], ;)

	        DEVICE_DEFINE(double, dpdf[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, dpdfd[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, dpdft[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, dpdfdt[IMAX][JMAX], ;)

	        DEVICE_DEFINE(double, ef[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, efd[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, eft[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, efdt[IMAX][JMAX], ;)

	        DEVICE_DEFINE(double, xf[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, xfd[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, xft[IMAX][JMAX], ;)
	        DEVICE_DEFINE(double, xfdt[IMAX][JMAX], ;)
		}
		bool debug = false;
	}
}