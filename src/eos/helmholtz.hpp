#pragma once

#include <iostream>

#include "../eigen.hpp"

#include <sstream>
#include <string>
#include <fstream>

#include <vector>
#include <tuple>
#include <math.h>

#ifndef IMAX
	#define IMAX 541
#endif
#ifndef JMAX
	#define JMAX 201
#endif


namespace nnet::eos {
	namespace helmholtz_constants {
		// table size
		const int imax = IMAX, jmax = JMAX;

		// table limits
		const double tlo   = 3.;
		const double thi   = 13.;
		const double tstp  = (thi - tlo)/(double)(jmax - 1);
		const double tstpi = 1./tstp;
		const double dlo   = -12.;
		const double dhi   = 15.;
		const double dstp  = (dhi - dlo)/(double)(imax - 1);
		const double dstpi = 1./dstp;

		// physical constants
		const double g       = 6.6742867e-8;
        const double h       = 6.6260689633e-27;
        const double hbar    = 0.5 * h/std::numbers::pi;
        const double qe      = 4.8032042712e-10;
        const double avo     = 6.0221417930e23;
        const double clight  = 2.99792458e10;
        const double kerg    = 1.380650424e-16;
        const double ev2erg  = 1.60217648740e-12;
        const double kev     = kerg/ev2erg;
        const double amu     = 1.66053878283e-24;
        const double mn      = 1.67492721184e-24;
        const double mp      = 1.67262163783e-24;
        const double me      = 9.1093821545e-28;
        const double rbohr   = hbar*hbar/(me * qe * qe);
        const double fine    = qe*qe/(hbar*clight);
        const double hion    = 13.605698140;
        const double ssol    = 5.6704e-5;
        const double asol    = 4.0 * ssol / clight;
        const double weinlam = h*clight/(kerg * 4.965114232);
        const double weinfre = 2.821439372*kerg/h;
        const double rhonuc  = 2.342e14;
        const double kergavo = kerg*avo;
		const double sioncon = (2.0*std::numbers::pi*amu*kerg)/(h*h);

		// tables
		double fi[36],

			d[imax], dd_sav[imax], dd2_sav[imax], ddi_sav[imax], dd2i_sav[imax], dd3i_sav[imax],
			t[jmax], dt_sav[jmax], dt2_sav[jmax], dti_sav[jmax], dt2i_sav[jmax], dt3i_sav[jmax],

	   		f[imax][jmax],
	   		fd[imax][jmax], ft[imax][jmax],
	   		fdd[imax][jmax], ftt[imax][jmax], fdt[imax][jmax],
	   		fddt[imax][jmax], fdtt[imax][jmax], fddtt[imax][jmax],

	   		dpdf[imax][jmax], dpdfd[imax][jmax], dpdft[imax][jmax], dpdfdt[imax][jmax],

	   		ef[imax][jmax], efd[imax][jmax], eft[imax][jmax], efdt[imax][jmax],

	   		xf[imax][jmax], xfd[imax][jmax], xft[imax][jmax], xfdt[imax][jmax];

		// read helmholtz constants table
		void read_table(const char *file_path){
	   		// read file
			std::ifstream helm_table; 
	   		helm_table.open(file_path);
	   		if (!helm_table) {
        		std::cerr << "Helm. table not found !\n";
        		throw;
	   		}

			// read the helmholtz free energy and its derivatives
			for (int i = 0; i < imax; ++i) {
				const double dsav = dlo + (i - 1)*dstp;
				d[i] = std::pow(10., dsav);
			}
			for (int j = 0; j < jmax; ++j) {
				const double tsav = tlo + (j - 1)*tstp;
				t[j] = std::pow(10., tsav);

				for (int i = 0; i < imax; ++i) {
					helm_table >> f[i][j] >> fd[i][j] >> ft[i][j] >>
			 			fdd[i][j] >> ftt[i][j] >> fdt[i][j] >>
			 			fddt[i][j] >> fdtt[i][j] >> fddtt[i][j];
				}
			}

			// read the pressure derivative with density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> dpdf[i][j] >> dpdfd[i][j] >> dpdft[i][j] >> dpdfdt[i][j];
				}

			// read the electron chemical potential table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> ef[i][j] >> efd[i][j] >> eft[i][j] >> efdt[i][j];
				}

			// read the number density table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> xf[i][j] >> xfd[i][j] >> xft[i][j] >> xfdt[i][j];
				}

			// construct the temperature and density deltas and their inverses
			for (int j = 0; j < jmax - 1; ++j) {
				const double dth  = t[j + 1] - t[j];
				const double dt2  = dth*dth;
				const double dti  = 1.0/dth;
				const double dt2i = 1.0/dt2;
				const double dt3i = dt2i*dti;

				dt_sav[j]   = dth;
				dt2_sav[j]  = dt2;
				dti_sav[j]  = dti;
				dt2i_sav[j] = dt2i;
				dt3i_sav[j] = dt3i;
			}

			// construct the temperature and density deltas and their inverses
			for (int i = 0; i < imax - 1; ++i) {
				const double dd   = d[i + 1] - d[i];
				const double dd2  = dd*dd;
				const double ddi  = 1.0/dd;
				const double dd2i = 1.0/dd2;
				const double dd3i = dd2i*ddi;

				dd_sav[i]   = dd;
				dd2_sav[i]  = dd2;
				ddi_sav[i]  = ddi;
				dd2i_sav[i] = dd2i;
				dd3i_sav[i] = dd3i;
			}

			helm_table.close();
		};


		// quintic hermite polynomial statement functions
		// psi0 and its derivatives
		auto const psi0 = [](const double z) {
			return z*z*z*(z*(-6.*z + 15.) - 10.) + 1.;
		};
		auto const dpsi0 = [](const double z) {
			return z*z*(z*(-30.*z + 60.) - 30.);
		};
		auto const ddpsi0 = [](const double z) {
			return z*(z*(-120.*z + 180.) -60.);
		};

		// psi1 and its derivatives
		auto const psi1 = [](const double z) {
			return z*(z*z*(z*(-3.*z + 8.) - 6.) + 1.);
		};
		auto const dpsi1 = [](const double z) {
			return z*z*(z*(-15.*z + 32.) - 18.) + 1.;
		};
		auto const ddpsi1 = [](const double z) {
			return z*(z*(-60.*z + 96.) -36.);
		};

		// psi2  and its derivatives
		auto const psi2 = [](const double z) {
			return 0.5*z*z*(z*(z*(-z + 3.) - 3.) + 1.);
		};
		auto const dpsi2 = [](const double z) {
			return  0.5*z*(z*(z*(-5.*z + 12.) - 9.) + 2.);
		};
		auto const ddpsi2 = [](const double z) {
			return 0.5*(z*(z*(-20.*z + 36.) - 18.) + 2.);
		};

		// biquintic hermite polynomial statement function
		auto const h5 = [](const double i, const double j,
			const double w0t, const double w1t, const double w2t, const double w0mt, const double w1mt, const double w2mt, const double w0d, const double w1d, const double w2d, const double w0md, const double w1md, const double w2md) {
		    return fi[1]*w0d*w0t  +  fi[2]*w0md*w0t
		    	+  fi[3]*w0d*w0mt +  fi[4]*w0md*w0mt
		    	+  fi[5]*w0d*w1t  +  fi[6]*w0md*w1t
		    	+  fi[7]*w0d*w1mt +  fi[8]*w0md*w1mt
		    	+  fi[9]*w0d*w2t  + fi[10]*w0md*w2t
		    	+ fi[11]*w0d*w2mt + fi[12]*w0md*w2mt
		    	+ fi[13]*w1d*w0t  + fi[14]*w1md*w0t
		    	+ fi[15]*w1d*w0mt + fi[16]*w1md*w0mt
		    	+ fi[17]*w2d*w0t  + fi[18]*w2md*w0t
		    	+ fi[19]*w2d*w0mt + fi[20]*w2md*w0mt
		    	+ fi[21]*w1d*w1t  + fi[22]*w1md*w1t
		    	+ fi[23]*w1d*w1mt + fi[24]*w1md*w1mt
		    	+ fi[25]*w2d*w1t  + fi[26]*w2md*w1t
		    	+ fi[27]*w2d*w1mt + fi[28]*w2md*w1mt
		    	+ fi[29]*w1d*w2t  + fi[30]*w1md*w2t
		    	+ fi[31]*w1d*w2mt + fi[32]*w1md*w2mt
		    	+ fi[33]*w2d*w2t  + fi[34]*w2md*w2t
		    	+ fi[35]*w2d*w2mt + fi[36]*w2md*w2mt;
		};


		// cubic hermite polynomial statement functions
		// psi0 and its derivatives
		auto const xpsi0 = [](const double z) {
			return z*z*(2.*z - 3.) + 1.0;
		};
		auto const xdpsi0 = [](const double z) {
			return z*(6.*z - 6.);
		};

		// psi1 & derivatives
		auto const xpsi1 = [](const double z) {
			return z*(z*(z - 2.) + 1.);
		};
		auto const xdpsi1 = [](const double z) {
			return z*(3.*z - 4.) + 1.;
		};

		// bicubic hermite polynomial statement function
		auto const h3 = [](const double i, const double j,
			const double w0t, const double w1t, const double w0mt, const double w1mt, const double w0d, const double w1d, const double w0md, const double w1md) {
		    return fi[1]*w0d*w0t  +  fi[2]*w0md*w0t
		    	+  fi[3]*w0d*w0mt +  fi[4]*w0md*w0mt
		    	+  fi[5]*w0d*w1t  +  fi[6]*w0md*w1t
		    	+  fi[7]*w0d*w1mt +  fi[8]*w0md*w1mt
		    	+  fi[9]*w1d*w0t  + fi[10]*w1md*w0t
		    	+ fi[11]*w1d*w0mt + fi[12]*w1md*w0mt
		    	+ fi[13]*w1d*w1t  + fi[14]*w1md*w1t
		    	+ fi[15]*w1d*w1mt + fi[16]*w1md*w1mt;
		};


		// get correspong table indices
		std::pair<int, int> get_table_indices(const double T, const double rho, const double abar, const double zbar) {
			const double ye = std::max(1e-16, zbar/abar);
			const double din = ye*rho;

			int jat = int((std::log10(T) - tlo)*tstpi) + 1;
			jat = std::max(1, std::min(jat, jmax - 1));

			int iat = int((std::log10(din) - dlo)*dstpi) + 1;
			iat = std::max(1, std::min(iat, imax - 1));

			return {jat, iat};
		}

		// move table values into coefficient table
		void move_polynomial_coefs(const int jat, const int iat) {
			// move table values into coefficient table
			fi[1]  = f[iat][jat];
			fi[2]  = f[iat + 1][jat];
			fi[3]  = f[iat][jat + 1];
			fi[4]  = f[iat + 1][jat + 1];
			fi[5]  = ft[iat][jat];
			fi[6]  = ft[iat + 1][jat];
			fi[7]  = ft[iat][jat + 1];
			fi[8]  = ft[iat + 1][jat + 1];
			fi[9]  = ftt[iat][jat];
			fi[10] = ftt[iat + 1][jat];
			fi[11] = ftt[iat][jat + 1];
			fi[12] = ftt[iat + 1][jat + 1];
			fi[13] = fd[iat][jat];
			fi[14] = fd[iat + 1][jat];
			fi[15] = fd[iat][jat + 1];
			fi[16] = fd[iat + 1][jat + 1];
			fi[17] = fdd[iat][jat];
			fi[18] = fdd[iat + 1][jat];
			fi[19] = fdd[iat][jat + 1];
			fi[20] = fdd[iat + 1][jat + 1];
			fi[21] = fdt[iat][jat];
			fi[22] = fdt[iat + 1][jat];
			fi[23] = fdt[iat][jat + 1];
			fi[24] = fdt[iat + 1][jat + 1];
			fi[25] = fddt[iat][jat];
			fi[26] = fddt[iat + 1][jat];
			fi[27] = fddt[iat][jat + 1];
			fi[28] = fddt[iat + 1][jat + 1];
			fi[29] = fdtt[iat][jat];
			fi[30] = fdtt[iat + 1][jat];
			fi[31] = fdtt[iat][jat + 1];
			fi[32] = fdtt[iat + 1][jat + 1];
			fi[33] = fddtt[iat][jat];
			fi[34] = fddtt[iat + 1][jat];
			fi[35] = fddtt[iat][jat + 1];
			fi[36] = fddtt[iat + 1][jat + 1];
		}
	}



	/// helmholtz eos
	/**
	 * ...TODO
	 */
	template<typename Float>
	struct helmholtz {
		std::vector<Float> A, Z;
		struct eos_output {
			Float cv, dP_dT, P; //...
		};

		eos_output operator()(const std::vector<Float> &Y, const Float T, const Float rho) {
			const int dimension = Y.size();

			// compute abar and zbar
			Float abar=0, zbar=0;
			for (int i = 0; i < dimension; ++i) {
				abar += Y[i];
				zbar += Y[i]*Z[i];
			}
			abar = 1/abar;
			zbar = abar*zbar;

			// compute polynoms rates
			auto const [jat, iat] = get_table_indices(T, rho, abar, zbar);
			move_polynomial_coefs(jat, iat);


			const Float ytot1 = 1/abar;

			// initialize
			const Float deni    = 1./rho;
			const Float tempi   = 1./T;
			const Float kt      = helmholtz_constants::kerg*T;
			const Float ktinv   = 1./kt;


			// adiation section:
			const Float prad    = helmholtz_constants::asol*T*T*T*T/3;
			const Float dpraddd = 0.;
			const Float dpraddt = 4.*prad*tempi;
			const Float dpradda = 0.;
			const Float dpraddz = 0.;

			const Float erad    = 3.*prad*deni;
			const Float deraddd = -erad*deni;
			const Float deraddt = 3.*dpraddt*deni;
			const Float deradda = 0.;
			const Float deraddz = 0.;

			const Float srad    = (prad*deni + erad)*tempi;
			const Float dsraddd = (dpraddd*deni - prad*deni*deni + deraddd)*tempi;
			const Float dsraddt = (dpraddt*deni + deraddt - srad)*tempi;
			const Float dsradda = 0.;
			const Float dsraddz = 0.;


			// ion section:
			const Float xni     = helmholtz_constants::avo*ytot1*rho;
			const Float dxnidd  = helmholtz_constants::avo*ytot1;
			const Float dxnida  = -xni*ytot1;

			const Float pion    = xni*kt;
			const Float dpiondd = dxnidd*kt;
			const Float dpiondt = xni*helmholtz_constants::kerg;
			const Float dpionda = dxnida*kt;
			const Float dpiondz = 0.;

			const Float eion    = 1.5*pion*deni;
			const Float deiondd = (1.5*dpiondd - eion)*deni;
			const Float deiondt = 1.5*dpiondt*deni;
			const Float deionda = 1.5*dpionda*deni;
			const Float deiondz = 0.;


			// sackur-tetrode equation for the ion entropy of
			// a single ideal gas characterized by abar
			      Float x = abar*abar*std::sqrt(abar) * deni/helmholtz_constants::avo;
			const Float s = helmholtz_constants::sioncon*T;
			const Float z = x*s*std::sqrt(s);
			const Float y = std::log(z);

			// y       = 1.0/(abar*kt)
			// yy      = y * sqrt(y)
			// z       = xni * sifac * yy
			// etaion  = log(z)


			const Float sion    = (pion*deni + eion)*tempi + helmholtz_constants::kergavo*ytot1*y;
			const Float dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - helmholtz_constants::kergavo*deni*ytot1;
			const Float dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion)*tempi*tempi + 1.5*helmholtz_constants::kergavo*tempi*ytot1;
			            x       = helmholtz_constants::avo*helmholtz_constants::kerg/abar;
			const Float dsionda = (dpionda*deni + deionda)*tempi + helmholtz_constants::kergavo*ytot1*ytot1*(2.5 - y);
			const Float dsiondz = 0.;



			// electron-positron section:


			// assume complete ionization
			const Float xnem    = xni * zbar;

			/* TODO... */
			eos_output res;
			res.cv = 2e7;
			res.dP_dT = 0;
			res.P = 0;

			return res;
		}
	};
}

