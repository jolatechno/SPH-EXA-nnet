#include "nnet/net87/net87.hpp"
#include "nnet/net86/net86.hpp"
#include "nnet/net14/net14.hpp"

#include "nnet/eos/helmholtz.hpp"
#include "nnet/eos/ideal_gas.hpp"

namespace nnet {
	namespace net87 {
		namespace electrons::constants {
			DEVICE_DEFINE_DETAIL(,, double, log_temp_ref[N_TEMP], ;)
	        DEVICE_DEFINE_DETAIL(,, double, log_rho_ref[N_RHO], ;)
	        DEVICE_DEFINE_DETAIL(,, double, electron_rate[N_TEMP][N_RHO][N_C], ;)
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
			DEVICE_DEFINE_DETAIL(,, double, d[IMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dd_sav[IMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dd2_sav[IMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, ddi_sav[IMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dd2i_sav[IMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dd3i_sav[IMAX - 1], ;)

	        DEVICE_DEFINE_DETAIL(,, double, t_[JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dt_sav[JMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dt2_sav[JMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dti_sav[JMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dt2i_sav[JMAX - 1], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dt3i_sav[JMAX - 1], ;)

	        DEVICE_DEFINE_DETAIL(,, double, f[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fd[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, ft[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fdd[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, ftt[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fdt[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fddt[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fdtt[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, fddtt[IMAX][JMAX], ;)

	        DEVICE_DEFINE_DETAIL(,, double, dpdf[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dpdfd[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dpdft[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, dpdfdt[IMAX][JMAX], ;)

	        DEVICE_DEFINE_DETAIL(,, double, ef[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, efd[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, eft[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, efdt[IMAX][JMAX], ;)

	        DEVICE_DEFINE_DETAIL(,, double, xf[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, xfd[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, xft[IMAX][JMAX], ;)
	        DEVICE_DEFINE_DETAIL(,, double, xfdt[IMAX][JMAX], ;)
		}
		bool debug = false;
	}
}