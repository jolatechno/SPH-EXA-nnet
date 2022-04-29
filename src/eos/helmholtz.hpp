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
			const double w0t, const double w1t, const double w2t, const double w0mt, const double w1mt, const double w2mt,
			const double w0d,const double w1d, const double w2d, const double w0md, const double w1md, const double w2md) {
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


			Float ytot1 = 1/abar;
			Float ye = std::max(1e-16, zbar/abar);
			Float din = ye*rho;

			// initialize
			Float deni    = 1./rho;
			Float tempi   = 1./T;
			Float kt      = helmholtz_constants::kerg*T;
			Float ktinv   = 1./kt;


			// adiation section:
			Float prad    = helmholtz_constants::asol*T*T*T*T/3;
			Float dpraddd = 0.;
			Float dpraddt = 4.*prad*tempi;
			Float dpradda = 0.;
			Float dpraddz = 0.;

			Float erad    = 3.*prad*deni;
			Float deraddd = -erad*deni;
			Float deraddt = 3.*dpraddt*deni;
			Float deradda = 0.;
			Float deraddz = 0.;

			Float srad    = (prad*deni + erad)*tempi;
			Float dsraddd = (dpraddd*deni - prad*deni*deni + deraddd)*tempi;
			Float dsraddt = (dpraddt*deni + deraddt - srad)*tempi;
			Float dsradda = 0.;
			Float dsraddz = 0.;


			// ion section:
			Float xni     = helmholtz_constants::avo*ytot1*rho;
			Float dxnidd  = helmholtz_constants::avo*ytot1;
			Float dxnida  = -xni*ytot1;

			Float pion    = xni*kt;
			Float dpiondd = dxnidd*kt;
			Float dpiondt = xni*helmholtz_constants::kerg;
			Float dpionda = dxnida*kt;
			Float dpiondz = 0.;

			Float eion    = 1.5*pion*deni;
			Float deiondd = (1.5*dpiondd - eion)*deni;
			Float deiondt = 1.5*dpiondt*deni;
			Float deionda = 1.5*dpionda*deni;
			Float deiondz = 0.;


			// sackur-tetrode equation for the ion entropy of
			// a single ideal gas characterized by abar
			      Float x = abar*abar*std::sqrt(abar) * deni/helmholtz_constants::avo;
			Float s = helmholtz_constants::sioncon*T;
			Float z = x*s*std::sqrt(s);
			Float y = std::log(z);

			// y       = 1.0/(abar*kt)
			// yy      = y * sqrt(y)
			// z       = xni * sifac * yy
			// etaion  = log(z)


			Float sion    = (pion*deni + eion)*tempi + helmholtz_constants::kergavo*ytot1*y;
			Float dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi - helmholtz_constants::kergavo*deni*ytot1;
			Float dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion)*tempi*tempi + 1.5*helmholtz_constants::kergavo*tempi*ytot1;
			            x       = helmholtz_constants::avo*helmholtz_constants::kerg/abar;
			Float dsionda = (dpionda*deni + deionda)*tempi + helmholtz_constants::kergavo*ytot1*ytot1*(2.5 - y);
			Float dsiondz = 0.;



			// electron-positron section:


			// assume complete ionization
			Float xnem    = xni * zbar;






			// move table values into coefficient table
			helmholtz_constants::fi[1]  = helmholtz_constants::f[iat][jat];
			helmholtz_constants::fi[2]  = helmholtz_constants::f[iat + 1][jat];
			helmholtz_constants::fi[3]  = helmholtz_constants::f[iat][jat + 1];
			helmholtz_constants::fi[4]  = helmholtz_constants::f[iat + 1][jat + 1];
			helmholtz_constants::fi[5]  = helmholtz_constants::ft[iat][jat];
			helmholtz_constants::fi[6]  = helmholtz_constants::ft[iat + 1][jat];
			helmholtz_constants::fi[7]  = helmholtz_constants::ft[iat][jat + 1];
			helmholtz_constants::fi[8]  = helmholtz_constants::ft[iat + 1][jat + 1];
			helmholtz_constants::fi[9]  = helmholtz_constants::ftt[iat][jat];
			helmholtz_constants::fi[10] = helmholtz_constants::ftt[iat + 1][jat];
			helmholtz_constants::fi[11] = helmholtz_constants::ftt[iat][jat + 1];
			helmholtz_constants::fi[12] = helmholtz_constants::ftt[iat + 1][jat + 1];
			helmholtz_constants::fi[13] = helmholtz_constants::fd[iat][jat];
			helmholtz_constants::fi[14] = helmholtz_constants::fd[iat + 1][jat];
			helmholtz_constants::fi[15] = helmholtz_constants::fd[iat][jat + 1];
			helmholtz_constants::fi[16] = helmholtz_constants::fd[iat + 1][jat + 1];
			helmholtz_constants::fi[17] = helmholtz_constants::fdd[iat][jat];
			helmholtz_constants::fi[18] = helmholtz_constants::fdd[iat + 1][jat];
			helmholtz_constants::fi[19] = helmholtz_constants::fdd[iat][jat + 1];
			helmholtz_constants::fi[20] = helmholtz_constants::fdd[iat + 1][jat + 1];
			helmholtz_constants::fi[21] = helmholtz_constants::fdt[iat][jat];
			helmholtz_constants::fi[22] = helmholtz_constants::fdt[iat + 1][jat];
			helmholtz_constants::fi[23] = helmholtz_constants::fdt[iat][jat + 1];
			helmholtz_constants::fi[24] = helmholtz_constants::fdt[iat + 1][jat + 1];
			helmholtz_constants::fi[25] = helmholtz_constants::fddt[iat][jat];
			helmholtz_constants::fi[26] = helmholtz_constants::fddt[iat + 1][jat];
			helmholtz_constants::fi[27] = helmholtz_constants::fddt[iat][jat + 1];
			helmholtz_constants::fi[28] = helmholtz_constants::fddt[iat + 1][jat + 1];
			helmholtz_constants::fi[29] = helmholtz_constants::fdtt[iat][jat];
			helmholtz_constants::fi[30] = helmholtz_constants::fdtt[iat + 1][jat];
			helmholtz_constants::fi[31] = helmholtz_constants::fdtt[iat][jat + 1];
			helmholtz_constants::fi[32] = helmholtz_constants::fdtt[iat + 1][jat + 1];
			helmholtz_constants::fi[33] = helmholtz_constants::fddtt[iat][jat];
			helmholtz_constants::fi[34] = helmholtz_constants::fddtt[iat + 1][jat];
			helmholtz_constants::fi[35] = helmholtz_constants::fddtt[iat][jat + 1];
			helmholtz_constants::fi[36] = helmholtz_constants::fddtt[iat + 1][jat + 1];




			// various differences
			Float xt  = std::max( (T - t(jat))*dti_sav(jat), 0.);
			Float xd  = std::max( (din - d(iat))*ddi_sav(iat), 0.);
			Float mxt = 1. - xt;
			Float mxd = 1. - xd;

			// the six density and six temperature basis functions;
			Float si0t =   helmholtz_constants::psi0(xt);
			Float si1t =   helmholtz_constants::psi1(xt)*dt_sav(jat);
			Float si2t =   helmholtz_constants::psi2(xt)*dt2_sav(jat);

			Float si0mt =  helmholtz_constants::psi0(mxt);
			Float si1mt = -helmholtz_constants::psi1(mxt)*dt_sav(jat);
			Float si2mt =  helmholtz_constants::psi2(mxt)*dt2_sav(jat);

			Float si0d =   helmholtz_constants::psi0(xd);
			Float si1d =   helmholtz_constants::psi1(xd)*dd_sav(iat);
			Float si2d =   helmholtz_constants::psi2(xd)*dd2_sav(iat);

			Float si0md =  helmholtz_constants::psi0(mxd);
			Float si1md = -helmholtz_constants::psi1(mxd)*dd_sav(iat);
			Float si2md =  helmholtz_constants::psi2(mxd)*dd2_sav(iat);

			// derivatives of the weight functions
			Float dsi0t =   helmholtz_constants::dpsi0(xt)*dti_sav(jat);
			Float dsi1t =   helmholtz_constants::dpsi1(xt);
			Float dsi2t =   helmholtz_constants::dpsi2(xt)*dt_sav(jat);

			Float dsi0mt = -helmholtz_constants::dpsi0(mxt)*dti_sav(jat);
			Float dsi1mt =  helmholtz_constants::dpsi1(mxt);
			Float dsi2mt = -helmholtz_constants::dpsi2(mxt)*dt_sav(jat);

			Float dsi0d =   helmholtz_constants::dpsi0(xd)*ddi_sav(iat);
			Float dsi1d =   helmholtz_constants::dpsi1(xd);
			Float dsi2d =   helmholtz_constants::dpsi2(xd)*dd_sav(iat);

			Float dsi0md = -helmholtz_constants::dpsi0(mxd)*ddi_sav(iat);
			Float dsi1md =  helmholtz_constants::dpsi1(mxd);
			Float dsi2md = -helmholtz_constants::dpsi2(mxd)*dd_sav(iat);

			// second derivatives of the weight functions
			Float ddsi0t =   helmholtz_constants::ddpsi0(xt)*dt2i_sav(jat);
			Float ddsi1t =   helmholtz_constants::ddpsi1(xt)*dti_sav(jat);
			Float ddsi2t =   helmholtz_constants::ddpsi2(xt);

			Float ddsi0mt =  helmholtz_constants::ddpsi0(mxt)*dt2i_sav(jat);
			Float ddsi1mt = -helmholtz_constants::ddpsi1(mxt)*dti_sav(jat);
			Float ddsi2mt =  helmholtz_constants::ddpsi2(mxt);

			// ddsi0d =   ddpsi0(xd)*dd2i_sav(iat);
			// ddsi1d =   ddpsi1(xd)*ddi_sav(iat);
			// ddsi2d =   ddpsi2(xd);

			// ddsi0md =  ddpsi0(mxd)*dd2i_sav(iat);
			// ddsi1md = -ddpsi1(mxd)*ddi_sav(iat);
			// ddsi2md =  ddpsi2(mxd);


			// the free energy
			Float free  = helmholtz_constants::h5(iat,jat,
				si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

			// derivative with respect to density
			Float df_d  = helmholtz_constants::h5(iat,jat,
				si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);


			// derivative with respect to temperature
			Float df_t = helmholtz_constants::h5(iat,jat,
				dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

			// derivative with respect to density**2
			// df_dd = h5(iat,jat,
			//		si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
			//		ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md)

			// derivative with respect to temperature**2
			Float df_tt = helmholtz_constants::h5(iat,jat,
				ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

			// derivative with respect to temperature and density
			Float df_dt = helmholtz_constants::h5(iat,jat,
				dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);



			// now get the pressure derivative with density, chemical potential, and
			// electron positron number densities
			// get the interpolation weight functions
			si0t   =  helmholtz_constants::xpsi0(xt);
			si1t   =  helmholtz_constants::xpsi1(xt)*dt_sav(jat);

			si0mt  =  helmholtz_constants::xpsi0(mxt);
			si1mt  =  -helmholtz_constants::xpsi1(mxt)*dt_sav(jat);

			si0d   =  helmholtz_constants::xpsi0(xd);
			si1d   =  helmholtz_constants::xpsi1(xd)*dd_sav(iat);

			si0md  =  helmholtz_constants::xpsi0(mxd);
			si1md  =  -helmholtz_constants::xpsi1(mxd)*dd_sav(iat);


			// derivatives of weight functions
			dsi0t  = helmholtz_constants::xdpsi0(xt)*dti_sav(jat);
			dsi1t  = helmholtz_constants::xdpsi1(xt);

			dsi0mt = -helmholtz_constants::xdpsi0(mxt)*dti_sav(jat);
			dsi1mt = helmholtz_constants::xdpsi1(mxt);

			dsi0d  = helmholtz_constants::xdpsi0(xd)*ddi_sav(iat);
			dsi1d  = helmholtz_constants::xdpsi1(xd);

			dsi0md = -helmholtz_constants::xdpsi0(mxd)*ddi_sav(iat);
			dsi1md = helmholtz_constants::xdpsi1(mxd);





			// move table values into coefficient table
			helmholtz_constants::fi[1]  = helmholtz_constants::dpdf[iat][jat];
			helmholtz_constants::fi[2]  = helmholtz_constants::dpdf[iat+1][jat];
			helmholtz_constants::fi[3]  = helmholtz_constants::dpdf[iat][jat+1];
			helmholtz_constants::fi[4]  = helmholtz_constants::dpdf[iat+1][jat+1];
			helmholtz_constants::fi[5]  = helmholtz_constants::dpdft[iat][jat];
			helmholtz_constants::fi[6]  = helmholtz_constants::dpdft[iat+1][jat];
			helmholtz_constants::fi[7]  = helmholtz_constants::dpdft[iat][jat+1];
			helmholtz_constants::fi[8]  = helmholtz_constants::dpdft[iat+1][jat+1];
			helmholtz_constants::fi[9]  = helmholtz_constants::dpdfd[iat][jat];
			helmholtz_constants::fi[10] = helmholtz_constants::dpdfd[iat+1][jat];
			helmholtz_constants::fi[11] = helmholtz_constants::dpdfd[iat][jat+1];
			helmholtz_constants::fi[12] = helmholtz_constants::dpdfd[iat+1][jat+1];
			helmholtz_constants::fi[13] = helmholtz_constants::dpdfdt[iat][jat];
			helmholtz_constants::fi[14] = helmholtz_constants::dpdfdt[iat+1][jat];
			helmholtz_constants::fi[15] = helmholtz_constants::dpdfdt[iat][jat+1];
			helmholtz_constants::fi[16] = helmholtz_constants::dpdfdt[iat+1][jat+1];




			Float dpepdd  = helmholtz_constants::h3(iat, jat,
                si0t,   si1t,   si0mt,   si1mt,
                si0d,   si1d,   si0md,   si1md);
  			dpepdd  = std::max(ye * dpepdd,1.e-30);


  			


			/* TODO... */
			eos_output res;
			res.cv = 2e7;
			res.dP_dT = 0;
			res.P = 0;

			return res;
		}
	};
}

