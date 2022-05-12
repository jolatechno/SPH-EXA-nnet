#pragma once

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

#include "../eigen.hpp"

#ifndef IMAX
	#define IMAX 541
#endif
#ifndef JMAX
	#define JMAX 201
#endif
#ifndef TABLE_PATH
	#define TABLE_PATH "./helm_table.dat"
#endif

#include <sstream>
#include <string>
#include <array>

#include <vector>
#include <tuple>
#include <math.h>

namespace nnet::eos {
	/* !!!!!!!!!!!!
	debuging :
	!!!!!!!!!!!! */
	bool debug = false;

	
	namespace helmholtz_constants {
		// table size
		const int imax = IMAX, jmax = JMAX;

		// table type
		typedef std::array<double, imax> ivector; // double[imax]
		typedef std::array<double, jmax> jvector; // double[jmax]
		typedef std::array<double, imax - 1> imvector; // double[imax]
		typedef std::array<double, jmax - 1> jmvector; // double[jmax]
		typedef eigen::fixed_size_matrix<double, imax, jmax> ijmatrix; // double(imax, jmax)

		// read table
		const std::string helmolt_table = { 
			#include TABLE_PATH
		};

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
        const double hbar    = 0.5*h/std::numbers::pi;
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
        const double rbohr   = hbar*hbar/(me*qe*qe);
        const double fine    = qe*qe/(hbar*clight);
        const double hion    = 13.605698140;
        const double ssol    = 5.6704e-5;
        const double asol    = 4.0*ssol / clight;
        const double weinlam = h*clight/(kerg*4.965114232);
        const double weinfre = 2.821439372*kerg/h;
        const double rhonuc  = 2.342e14;
        const double kergavo = kerg*avo;
		const double sioncon = (2.0*std::numbers::pi*amu*kerg)/(h*h);

		// parameters
		const double a1    = -0.898004;
        const double b1    =  0.96786;
        const double c1    =  0.220703;
        const double d1    = -0.86097;
        const double e1    =  2.5269;
        const double a2    =  0.29561;
        const double b2    =  1.9885;
        const double c2    =  0.288675;
        const double esqu  =  qe*qe;

		// read helmholtz constants table
		std::tuple<
				ivector,
				imvector, imvector, imvector, imvector, imvector,
				jvector,
				jmvector, jmvector, jmvector, jmvector, jmvector,

				ijmatrix,
				ijmatrix, ijmatrix,

				ijmatrix, ijmatrix, ijmatrix,
				ijmatrix, ijmatrix, ijmatrix,

				ijmatrix, ijmatrix, ijmatrix, ijmatrix,

				ijmatrix, ijmatrix, ijmatrix, ijmatrix,
				ijmatrix, ijmatrix, ijmatrix, ijmatrix
			> read_table(){
	   		// read file
	   		std::stringstream helm_table;
	   		helm_table << helmolt_table;

	   		// define tables
	   		ivector d;
	   		imvector dd_sav, dd2_sav, ddi_sav, dd2i_sav, dd3i_sav ;
	   		jvector t;
	   		jmvector dt_sav, dt2_sav, dti_sav, dt2i_sav, dt3i_sav ;
	   		ijmatrix f,
	   			fd, ft,
	   			fdd, ftt, fdt,
	   			fddt, fdtt, fddtt,

	   			dpdf, dpdfd, dpdft, dpdfdt,

	   			ef, efd, eft, efdt,

	   			xf, xfd, xft, xfdt;

			// read the helmholtz free energy and its derivatives
			for (int i = 0; i < imax; ++i) {
				const double dsav = dlo + i*dstp;
				d[i] = std::pow(10., dsav);
			}
			for (int j = 0; j < jmax; ++j) {
				const double tsav = tlo + j*tstp;
				t[j] = std::pow(10., tsav);

				for (int i = 0; i < imax; ++i) {
					helm_table >> f(i, j) >> fd(i, j) >> ft(i, j) >>
			 			fdd(i, j) >> ftt(i, j) >> fdt(i, j) >>
			 			fddt(i, j) >> fdtt(i, j) >> fddtt(i, j);
				}
			}

			// read the pressure derivative with rhosity table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> dpdf(i, j) >> dpdfd(i, j) >> dpdft(i, j) >> dpdfdt(i, j);
				}

			// read the electron chemical potential table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> ef(i, j) >> efd(i, j) >> eft(i, j) >> efdt(i, j);
				}

			// read the number rhosity table
			for (int j = 0; j < jmax; ++j)
				for (int i = 0; i < imax; ++i) {
					helm_table >> xf(i, j) >> xfd(i, j) >> xft(i, j) >> xfdt(i, j);
				}

			// construct the temperature and rhosity deltas and their inverses
			for (int j = 0; j < jmax - 1; ++j) {
				const double dth  = t[j + 1] - t[j];
				const double dt2  = dth*dth;
				const double dti  = 1./dth;
				const double dt2i = 1./dt2;
				const double dt3i = dt2i*dti;

				dt_sav[j]   = dth;
				dt2_sav[j]  = dt2;
				dti_sav[j]  = dti;
				dt2i_sav[j] = dt2i;
				dt3i_sav[j] = dt3i;
			}

			// construct the temperature and rhosity deltas and their inverses
			for (int i = 0; i < imax - 1; ++i) {
				const double dd   = d[i + 1] - d[i];
				const double dd2  = dd*dd;
				const double ddi  = 1./dd;
				const double dd2i = 1./dd2;
				const double dd3i = dd2i*ddi;

				dd_sav[i]   = dd;
				dd2_sav[i]  = dd2;
				ddi_sav[i]  = ddi;
				dd2i_sav[i] = dd2i;
				dd3i_sav[i] = dd3i;
			}

			return {
				d, dd_sav, dd2_sav, ddi_sav, dd2i_sav, dd3i_sav,
				t, dt_sav, dt2_sav, dti_sav, dt2i_sav, dt3i_sav,
				
				f,
	   			fd, ft,
	   			fdd, ftt, fdt,
	   			fddt, fdtt, fddtt,

	   			dpdf, dpdfd, dpdft, dpdfdt,

	   			ef, efd, eft, efdt,
	   			xf, xfd, xft, xfdt
	   		};
		};


		// tables
		double fi[36];
		auto const [
			d, dd_sav, dd2_sav, ddi_sav, dd2i_sav, dd3i_sav,
			t, dt_sav, dt2_sav, dti_sav, dt2i_sav, dt3i_sav,
			
			f,
   			fd, ft,
   			fdd, ftt, fdt,
   			fddt, fdtt, fddtt,

   			dpdf, dpdfd, dpdft, dpdfdt,

   			ef, efd, eft, efdt,
   			xf, xfd, xft, xfdt
	   	] = read_table();


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
		auto const h5 = [](const double w0t, const double w1t, const double w2t, const double w0mt, const double w1mt, const double w2mt,
			const double w0d,const double w1d, const double w2d, const double w0md, const double w1md, const double w2md) {
		    return fi[0]*w0d*w0t  +  fi[1]*w0md*w0t
		    	+  fi[2]*w0d*w0mt +  fi[3]*w0md*w0mt
		    	+  fi[4]*w0d*w1t  +  fi[5]*w0md*w1t
		    	+  fi[6]*w0d*w1mt +  fi[7]*w0md*w1mt
		    	+  fi[8]*w0d*w2t  +  fi[9]*w0md*w2t
		    	+ fi[10]*w0d*w2mt + fi[11]*w0md*w2mt
		    	+ fi[12]*w1d*w0t  + fi[13]*w1md*w0t
		    	+ fi[14]*w1d*w0mt + fi[15]*w1md*w0mt
		    	+ fi[16]*w2d*w0t  + fi[17]*w2md*w0t
		    	+ fi[18]*w2d*w0mt + fi[19]*w2md*w0mt
		    	+ fi[20]*w1d*w1t  + fi[21]*w1md*w1t
		    	+ fi[22]*w1d*w1mt + fi[23]*w1md*w1mt
		    	+ fi[24]*w2d*w1t  + fi[25]*w2md*w1t
		    	+ fi[26]*w2d*w1mt + fi[27]*w2md*w1mt
		    	+ fi[28]*w1d*w2t  + fi[29]*w1md*w2t
		    	+ fi[30]*w1d*w2mt + fi[31]*w1md*w2mt
		    	+ fi[32]*w2d*w2t  + fi[33]*w2md*w2t
		    	+ fi[34]*w2d*w2mt + fi[35]*w2md*w2mt;
		};


		// cubic hermite polynomial statement functions
		// psi0 and its derivatives
		auto const xpsi0 = [](const double z) {
			return z*z*(2.*z - 3.) + 1.;
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
		auto const h3 = [](const double w0t, const double w1t, const double w0mt, const double w1mt, const double w0d, const double w1d, const double w0md, const double w1md) {
		    return fi[0]*w0d*w0t  +  fi[1]*w0md*w0t
		    	+  fi[2]*w0d*w0mt +  fi[3]*w0md*w0mt
		    	+  fi[4]*w0d*w1t  +  fi[5]*w0md*w1t
		    	+  fi[6]*w0d*w1mt +  fi[7]*w0md*w1mt
		    	+  fi[8]*w1d*w0t  +  fi[9]*w1md*w0t
		    	+ fi[10]*w1d*w0mt + fi[11]*w1md*w0mt
		    	+ fi[12]*w1d*w1t  + fi[13]*w1md*w1t
		    	+ fi[14]*w1d*w1mt + fi[15]*w1md*w1mt;
		};


		// get correspong table indices
		std::pair<int, int> get_table_indices(const double T, const double rho, const double abar, const double zbar) {
			const double ye = std::max(1e-16, zbar/abar);
			const double din = ye*rho;

			int jat = int((std::log10(T) - tlo)*tstpi);
			jat = std::max(0, std::min(jat, jmax - 2));

			int iat = int((std::log10(din) - dlo)*dstpi);
			iat = std::max(0, std::min(iat, imax - 2));

			return {jat, iat};
		}
	}



	/// helmholtz eos
	/**
	*...TODO
	 */
	template<typename Float>
	class helmholtz {
	private:
		std::vector<Float> Z;

	public:
		helmholtz(const std::vector<Float> &Z_) : Z(Z_) {}
		auto operator()(const std::vector<Float> &Y, const Float T, const Float rho) const {
			const int dimension = Y.size();

			// compute abar and zbar
			Float abar=0, zbar=0;
			for (int i = 0; i < dimension; ++i) {
				abar += Y[i];
				zbar += Y[i]*Z[i];
			}
			abar = 1/abar;
			zbar = abar*zbar;


			/* debug: */
			if (debug) std::cout << "T=" << T << ", rho=" << rho << ", abar=" << abar << ", zbar=" << zbar << "\n";


			// compute polynoms rates
			auto const [jat, iat] = helmholtz_constants::get_table_indices(T, rho, abar, zbar);


			Float ytot1 = 1/abar;
			Float ye = std::max(1e-16, zbar/abar);
			Float din = ye*rho;

			// initialize
			Float rhoi    = 1./rho;
			Float tempi   = 1./T;
			Float kt      = helmholtz_constants::kerg*T;
			Float ktinv   = 1./kt;


			// adiation section:
			Float prad    = helmholtz_constants::asol*T*T*T*T/3;
			Float dpraddd = 0.;
			Float dpraddt = 4.*prad*tempi;
			Float dpradda = 0.;
			Float dpraddz = 0.;

			Float erad    = 3.*prad*rhoi;
			Float deraddd = -erad*rhoi;
			Float deraddt = 3.*dpraddt*rhoi;
			Float deradda = 0.;
			Float deraddz = 0.;

			

			Float srad    = (prad*rhoi + erad)*tempi;
			Float dsraddd = (dpraddd*rhoi - prad*rhoi*rhoi + deraddd)*tempi;
			Float dsraddt = (dpraddt*rhoi + deraddt - srad)*tempi;
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

			Float eion    = 1.5*pion*rhoi;
			Float deiondd = (1.5*dpiondd - eion)*rhoi;
			Float deiondt = 1.5*dpiondt*rhoi;
			Float deionda = 1.5*dpionda*rhoi;
			Float deiondz = 0.;


			// sackur-tetrode equation for the ion entropy of
			// a single ideal gas characterized by abar
			      Float x = abar*abar*std::sqrt(abar)*rhoi/helmholtz_constants::avo;
			Float s = helmholtz_constants::sioncon*T;
			Float z = x*s*std::sqrt(s);
			Float y = std::log(z);

			// y       = 1./(abar*kt)
			// yy      = y*sqrt(y)
			// z       = xni*sifac*yy
			// etaion  = log(z)


			Float sion    = (pion*rhoi + eion)*tempi + helmholtz_constants::kergavo*ytot1*y;
			Float dsiondd = (dpiondd*rhoi - pion*rhoi*rhoi + deiondd)*tempi - helmholtz_constants::kergavo*rhoi*ytot1;
			Float dsiondt = (dpiondt*rhoi + deiondt)*tempi - (pion*rhoi + eion)*tempi*tempi + 1.5*helmholtz_constants::kergavo*tempi*ytot1;
			            x       = helmholtz_constants::avo*helmholtz_constants::kerg/abar;
			Float dsionda = (dpionda*rhoi + deionda)*tempi + helmholtz_constants::kergavo*ytot1*ytot1*(2.5 - y);
			Float dsiondz = 0.;



			// electron-positron section:


			// assume complete ionization
			Float xnem    = xni*zbar;






			// move table values into coefficient table
			helmholtz_constants::fi[0]  = helmholtz_constants::f(iat + 0, jat + 0);
			helmholtz_constants::fi[1]  = helmholtz_constants::f(iat + 1, jat + 0);
			helmholtz_constants::fi[2]  = helmholtz_constants::f(iat + 0, jat + 1);
			helmholtz_constants::fi[3]  = helmholtz_constants::f(iat + 1, jat + 1);
			helmholtz_constants::fi[4]  = helmholtz_constants::ft(iat + 0, jat + 0);
			helmholtz_constants::fi[5]  = helmholtz_constants::ft(iat + 1, jat + 0);
			helmholtz_constants::fi[6]  = helmholtz_constants::ft(iat + 0, jat + 1);
			helmholtz_constants::fi[7]  = helmholtz_constants::ft(iat + 1, jat + 1);
			helmholtz_constants::fi[8]  = helmholtz_constants::ftt(iat + 0, jat + 0);
			helmholtz_constants::fi[9]  = helmholtz_constants::ftt(iat + 1, jat + 0);
			helmholtz_constants::fi[10] = helmholtz_constants::ftt(iat + 0, jat + 1);
			helmholtz_constants::fi[11] = helmholtz_constants::ftt(iat + 1, jat + 1);
			helmholtz_constants::fi[12] = helmholtz_constants::fd(iat + 0, jat + 0);
			helmholtz_constants::fi[13] = helmholtz_constants::fd(iat + 1, jat + 0);
			helmholtz_constants::fi[14] = helmholtz_constants::fd(iat + 0, jat + 1);
			helmholtz_constants::fi[15] = helmholtz_constants::fd(iat + 1, jat + 1);
			helmholtz_constants::fi[16] = helmholtz_constants::fdd(iat + 0, jat + 0);
			helmholtz_constants::fi[17] = helmholtz_constants::fdd(iat + 1, jat + 0);
			helmholtz_constants::fi[18] = helmholtz_constants::fdd(iat + 0, jat + 1);
			helmholtz_constants::fi[19] = helmholtz_constants::fdd(iat + 1, jat + 1);
			helmholtz_constants::fi[20] = helmholtz_constants::fdt(iat + 0, jat + 0);
			helmholtz_constants::fi[21] = helmholtz_constants::fdt(iat + 1, jat + 0);
			helmholtz_constants::fi[22] = helmholtz_constants::fdt(iat + 0, jat + 1);
			helmholtz_constants::fi[23] = helmholtz_constants::fdt(iat + 1, jat + 1);
			helmholtz_constants::fi[24] = helmholtz_constants::fddt(iat + 0, jat + 0);
			helmholtz_constants::fi[25] = helmholtz_constants::fddt(iat + 1, jat + 0);
			helmholtz_constants::fi[26] = helmholtz_constants::fddt(iat + 0, jat + 1);
			helmholtz_constants::fi[27] = helmholtz_constants::fddt(iat + 1, jat + 1);
			helmholtz_constants::fi[28] = helmholtz_constants::fdtt(iat + 0, jat + 0);
			helmholtz_constants::fi[29] = helmholtz_constants::fdtt(iat + 1, jat + 0);
			helmholtz_constants::fi[30] = helmholtz_constants::fdtt(iat + 0, jat + 1);
			helmholtz_constants::fi[31] = helmholtz_constants::fdtt(iat + 1, jat + 1);
			helmholtz_constants::fi[32] = helmholtz_constants::fddtt(iat + 0, jat + 0);
			helmholtz_constants::fi[33] = helmholtz_constants::fddtt(iat + 1, jat + 0);
			helmholtz_constants::fi[34] = helmholtz_constants::fddtt(iat + 0, jat + 1);
			helmholtz_constants::fi[35] = helmholtz_constants::fddtt(iat + 1, jat + 1);




			/* debug: */
			if (debug) {
				std::cout << "	fi[" << 0 << "]=" <<  helmholtz_constants::f(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 1 << "]=" <<  helmholtz_constants::f(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 2 << "]=" <<  helmholtz_constants::f(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 3 << "]=" <<  helmholtz_constants::f(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 4 << "]=" <<  helmholtz_constants::ft(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 5 << "]=" <<  helmholtz_constants::ft(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 6 << "]=" <<  helmholtz_constants::ft(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 7 << "]=" <<  helmholtz_constants::ft(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 8 << "]=" <<  helmholtz_constants::ftt(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 9 << "]=" <<  helmholtz_constants::ftt(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 10 << "]=" <<  helmholtz_constants::ftt(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 11 << "]=" <<  helmholtz_constants::ftt(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 12 << "]=" <<  helmholtz_constants::fd(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 13 << "]=" <<  helmholtz_constants::fd(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 14 << "]=" <<  helmholtz_constants::fd(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 15 << "]=" <<  helmholtz_constants::fd(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 16 << "]=" <<  helmholtz_constants::fdd(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 17 << "]=" <<  helmholtz_constants::fdd(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 18 << "]=" <<  helmholtz_constants::fdd(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 19 << "]=" <<  helmholtz_constants::fdd(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 20 << "]=" <<  helmholtz_constants::fdt(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 21 << "]=" <<  helmholtz_constants::fdt(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 22 << "]=" <<  helmholtz_constants::fdt(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 23 << "]=" <<  helmholtz_constants::fdt(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 24 << "]=" <<  helmholtz_constants::fddt(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 25 << "]=" <<  helmholtz_constants::fddt(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 26 << "]=" <<  helmholtz_constants::fddt(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 27 << "]=" <<  helmholtz_constants::fddt(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 28 << "]=" <<  helmholtz_constants::fdtt(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 29 << "]=" <<  helmholtz_constants::fdtt(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 30 << "]=" <<  helmholtz_constants::fdtt(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 31 << "]=" <<  helmholtz_constants::fdtt(iat + 1, jat + 1) << "\n";
				std::cout << "	fi[" << 32 << "]=" <<  helmholtz_constants::fddtt(iat + 0, jat + 0) << "\n";
				std::cout << "	fi[" << 33 << "]=" <<  helmholtz_constants::fddtt(iat + 1, jat + 0) << "\n";
				std::cout << "	fi[" << 34 << "]=" <<  helmholtz_constants::fddtt(iat + 0, jat + 1) << "\n";
				std::cout << "	fi[" << 35 << "]=" <<  helmholtz_constants::fddtt(iat + 1, jat + 1) << "\n";
			}




			// various differences
			Float xt  = std::max( (T - helmholtz_constants::t[jat])*helmholtz_constants::dti_sav[jat], 0.);
			Float xd  = std::max( (din - helmholtz_constants::d[iat])*helmholtz_constants::ddi_sav[iat], 0.);
			Float mxt = 1. - xt;
			Float mxd = 1. - xd;


			/* debug: */
			if (debug) std::cout << "xt=" << xt << " = (T - t[" << jat << "]=" << helmholtz_constants::t[jat] << ")* dti_sav[" << jat << "]=" << helmholtz_constants::dti_sav[jat] << "\n";


			// the six rhosity and six temperature basis functions;
			Float si0t =   helmholtz_constants::psi0(xt);
			Float si1t =   helmholtz_constants::psi1(xt)*helmholtz_constants::dt_sav[jat];
			Float si2t =   helmholtz_constants::psi2(xt)*helmholtz_constants::dt2_sav[jat];


			/* debug: */
			if (debug) std::cout << "si0t=" << si0t << " = psi0(xt=" << xt << ")\n";


			Float si0mt =  helmholtz_constants::psi0(mxt);
			Float si1mt = -helmholtz_constants::psi1(mxt)*helmholtz_constants::dt_sav[jat];
			Float si2mt =  helmholtz_constants::psi2(mxt)*helmholtz_constants::dt2_sav[jat];

			Float si0d =   helmholtz_constants::psi0(xd);
			Float si1d =   helmholtz_constants::psi1(xd)*helmholtz_constants::dd_sav[iat];
			Float si2d =   helmholtz_constants::psi2(xd)*helmholtz_constants::dd2_sav[iat];

			Float si0md =  helmholtz_constants::psi0(mxd);
			Float si1md = -helmholtz_constants::psi1(mxd)*helmholtz_constants::dd_sav[iat];
			Float si2md =  helmholtz_constants::psi2(mxd)*helmholtz_constants::dd2_sav[iat];

			// derivatives of the weight functions
			Float dsi0t =   helmholtz_constants::dpsi0(xt)*helmholtz_constants::dti_sav[jat];
			Float dsi1t =   helmholtz_constants::dpsi1(xt);
			Float dsi2t =   helmholtz_constants::dpsi2(xt)*helmholtz_constants::dt_sav[jat];

			Float dsi0mt = -helmholtz_constants::dpsi0(mxt)*helmholtz_constants::dti_sav[jat];
			Float dsi1mt =  helmholtz_constants::dpsi1(mxt);
			Float dsi2mt = -helmholtz_constants::dpsi2(mxt)*helmholtz_constants::dt_sav[jat];

			Float dsi0d =   helmholtz_constants::dpsi0(xd)*helmholtz_constants::ddi_sav[iat];
			Float dsi1d =   helmholtz_constants::dpsi1(xd);
			Float dsi2d =   helmholtz_constants::dpsi2(xd)*helmholtz_constants::dd_sav[iat];

			Float dsi0md = -helmholtz_constants::dpsi0(mxd)*helmholtz_constants::ddi_sav[iat];
			Float dsi1md =  helmholtz_constants::dpsi1(mxd);
			Float dsi2md = -helmholtz_constants::dpsi2(mxd)*helmholtz_constants::dd_sav[iat];

			// second derivatives of the weight functions
			Float ddsi0t =   helmholtz_constants::ddpsi0(xt)*helmholtz_constants::dt2i_sav[jat];
			Float ddsi1t =   helmholtz_constants::ddpsi1(xt)*helmholtz_constants::dti_sav[jat];
			Float ddsi2t =   helmholtz_constants::ddpsi2(xt);

			Float ddsi0mt =  helmholtz_constants::ddpsi0(mxt)*helmholtz_constants::dt2i_sav[jat];
			Float ddsi1mt = -helmholtz_constants::ddpsi1(mxt)*helmholtz_constants::dti_sav[jat];
			Float ddsi2mt =  helmholtz_constants::ddpsi2(mxt);

			// ddsi0d =   ddpsi0(xd)*dd2i_sav[iat];
			// ddsi1d =   ddpsi1(xd)*ddi_sav[iat];
			// ddsi2d =   ddpsi2(xd);

			// ddsi0md =  ddpsi0(mxd)*dd2i_sav[iat];
			// ddsi1md = -ddpsi1(mxd)*ddi_sav[iat];
			// ddsi2md =  ddpsi2(mxd);


			// the free energy
			Float free  = helmholtz_constants::h5(si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

			// derivative with respect to rhosity
			Float df_d  = helmholtz_constants::h5(si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);


			// derivative with respect to temperature
			Float df_t = helmholtz_constants::h5(dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

			// derivative with respect to rhosity**2
			// df_dd = h5(iat,jat,
			//		si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
			//		ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md)

			// derivative with respect to temperature**2
			Float df_tt = helmholtz_constants::h5(ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md);



			/* debug: */
			if (debug) std::cout << "df_tt=" << df_tt << " = h5(" << iat << ", " << jat << ",\n\t" <<
				ddsi0t << ", " << ddsi1t << ", " << ddsi2t << ", " << ddsi0mt << ", " << ddsi1mt << ", " << ddsi2mt << ",\n\t" <<
				si0d << ", " << si1d << ", " << si2d << ", " << si0md << ", " << si1md << ", " << si2md << ")\n";



			// derivative with respect to temperature and rhosity
			Float df_dt = helmholtz_constants::h5(dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);



			// now get the pressure derivative with rhosity, chemical potential, and
			// electron positron number rhosities
			// get the interpolation weight functions
			si0t   =  helmholtz_constants::xpsi0(xt);
			si1t   =  helmholtz_constants::xpsi1(xt)*helmholtz_constants::dt_sav[jat];

			si0mt  =  helmholtz_constants::xpsi0(mxt);
			si1mt  =  -helmholtz_constants::xpsi1(mxt)*helmholtz_constants::dt_sav[jat];

			si0d   =  helmholtz_constants::xpsi0(xd);
			si1d   =  helmholtz_constants::xpsi1(xd)*helmholtz_constants::dd_sav[iat];

			si0md  =  helmholtz_constants::xpsi0(mxd);
			si1md  =  -helmholtz_constants::xpsi1(mxd)*helmholtz_constants::dd_sav[iat];


			// derivatives of weight functions
			dsi0t  = helmholtz_constants::xdpsi0(xt)*helmholtz_constants::dti_sav[jat];
			dsi1t  = helmholtz_constants::xdpsi1(xt);

			dsi0mt = -helmholtz_constants::xdpsi0(mxt)*helmholtz_constants::dti_sav[jat];
			dsi1mt = helmholtz_constants::xdpsi1(mxt);

			dsi0d  = helmholtz_constants::xdpsi0(xd)*helmholtz_constants::ddi_sav[iat];
			dsi1d  = helmholtz_constants::xdpsi1(xd);

			dsi0md = -helmholtz_constants::xdpsi0(mxd)*helmholtz_constants::ddi_sav[iat];
			dsi1md = helmholtz_constants::xdpsi1(mxd);





			// move table values into coefficient table
			helmholtz_constants::fi[0]  = helmholtz_constants::dpdf(iat + 0, jat + 0);
			helmholtz_constants::fi[1]  = helmholtz_constants::dpdf(iat + 1, jat + 0);
			helmholtz_constants::fi[2]  = helmholtz_constants::dpdf(iat + 0, jat + 1);
			helmholtz_constants::fi[3]  = helmholtz_constants::dpdf(iat + 1, jat + 1);
			helmholtz_constants::fi[4]  = helmholtz_constants::dpdft(iat + 0, jat + 0);
			helmholtz_constants::fi[5]  = helmholtz_constants::dpdft(iat + 1, jat + 0);
			helmholtz_constants::fi[6]  = helmholtz_constants::dpdft(iat + 0, jat + 1);
			helmholtz_constants::fi[7]  = helmholtz_constants::dpdft(iat + 1, jat + 1);
			helmholtz_constants::fi[8]  = helmholtz_constants::dpdfd(iat + 0, jat + 0);
			helmholtz_constants::fi[9]  = helmholtz_constants::dpdfd(iat + 1, jat + 0);
			helmholtz_constants::fi[10] = helmholtz_constants::dpdfd(iat + 0, jat + 1);
			helmholtz_constants::fi[11] = helmholtz_constants::dpdfd(iat + 1, jat + 1);
			helmholtz_constants::fi[12] = helmholtz_constants::dpdfdt(iat + 0, jat + 0);
			helmholtz_constants::fi[13] = helmholtz_constants::dpdfdt(iat + 1, jat + 0);
			helmholtz_constants::fi[14] = helmholtz_constants::dpdfdt(iat + 0, jat + 1);
			helmholtz_constants::fi[15] = helmholtz_constants::dpdfdt(iat + 1, jat + 1);




			Float dpepdd  = helmholtz_constants::h3(si0t,   si1t,   si0mt,   si1mt,
                si0d,   si1d,   si0md,   si1md);
  			dpepdd  = std::max(ye*dpepdd, 1.e-30);





			// move table values into coefficient table
			helmholtz_constants::fi[0]  = helmholtz_constants::ef(iat + 0, jat + 0);
			helmholtz_constants::fi[1]  = helmholtz_constants::ef(iat + 1, jat + 0);
			helmholtz_constants::fi[2]  = helmholtz_constants::ef(iat + 0, jat + 1);
			helmholtz_constants::fi[3]  = helmholtz_constants::ef(iat + 1, jat + 1);
			helmholtz_constants::fi[4]  = helmholtz_constants::eft(iat + 0, jat + 0);
			helmholtz_constants::fi[5]  = helmholtz_constants::eft(iat + 1, jat + 0);
			helmholtz_constants::fi[6]  = helmholtz_constants::eft(iat + 0, jat + 1);
			helmholtz_constants::fi[7]  = helmholtz_constants::eft(iat + 1, jat + 1);
			helmholtz_constants::fi[8]  = helmholtz_constants::efd(iat + 0, jat + 0);
			helmholtz_constants::fi[9]  = helmholtz_constants::efd(iat + 1, jat + 0);
			helmholtz_constants::fi[10] = helmholtz_constants::efd(iat + 0, jat + 1);
			helmholtz_constants::fi[11] = helmholtz_constants::efd(iat + 1, jat + 1);
			helmholtz_constants::fi[12] = helmholtz_constants::efdt(iat + 0, jat + 0);
			helmholtz_constants::fi[13] = helmholtz_constants::efdt(iat + 1, jat + 0);
			helmholtz_constants::fi[14] = helmholtz_constants::efdt(iat + 0, jat + 1);
			helmholtz_constants::fi[15] = helmholtz_constants::efdt(iat + 1, jat + 1);






			// electron chemical potential etaele
			Float etaele  = helmholtz_constants::h3(si0t,   si1t,   si0mt,   si1mt,
				si0d,   si1d,   si0md,   si1md);


			// derivative with respect to rhosity
			x = helmholtz_constants::h3(si0t,   si1t,   si0mt,   si1mt,
				dsi0d,  dsi1d,  dsi0md,  dsi1md);
			Float detadd  = ye*x;

			// derivative with respect to temperature
			Float detadt  = helmholtz_constants::h3(dsi0t,  dsi1t,  dsi0mt,  dsi1mt,
				si0d,   si1d,   si0md,   si1md);

			// derivative with respect to abar and zbar
			Float detada = -x*din*ytot1;
			Float detadz =  x*rho*ytot1;





			// move table values into coefficient table
			helmholtz_constants::fi[0]  = helmholtz_constants::xf(iat + 0, jat + 0);
			helmholtz_constants::fi[1]  = helmholtz_constants::xf(iat + 1, jat + 0);
			helmholtz_constants::fi[2]  = helmholtz_constants::xf(iat + 0, jat + 1);
			helmholtz_constants::fi[3]  = helmholtz_constants::xf(iat + 1, jat + 1);
			helmholtz_constants::fi[4]  = helmholtz_constants::xft(iat + 0, jat + 0);
			helmholtz_constants::fi[5]  = helmholtz_constants::xft(iat + 1, jat + 0);
			helmholtz_constants::fi[6]  = helmholtz_constants::xft(iat + 0, jat + 1);
			helmholtz_constants::fi[7]  = helmholtz_constants::xft(iat + 1, jat + 1);
			helmholtz_constants::fi[8]  = helmholtz_constants::xfd(iat + 0, jat + 0);
			helmholtz_constants::fi[9]  = helmholtz_constants::xfd(iat + 1, jat + 0);
			helmholtz_constants::fi[10] = helmholtz_constants::xfd(iat + 0, jat + 1);
			helmholtz_constants::fi[11] = helmholtz_constants::xfd(iat + 1, jat + 1);
			helmholtz_constants::fi[12] = helmholtz_constants::xfdt(iat + 0, jat + 0);
			helmholtz_constants::fi[13] = helmholtz_constants::xfdt(iat + 1, jat + 0);
			helmholtz_constants::fi[14] = helmholtz_constants::xfdt(iat + 0, jat + 1);
			helmholtz_constants::fi[15] = helmholtz_constants::xfdt(iat + 1, jat + 1);





			// electron + positron number rhosities
			Float xnefer = helmholtz_constants::h3(si0t,   si1t,   si0mt,   si1mt,
            	si0d,   si1d,   si0md,   si1md);

			// derivative with respect to rhosity
			x = helmholtz_constants::h3(si0t,   si1t,   si0mt,   si1mt,
            	dsi0d,  dsi1d,  dsi0md,  dsi1md);
			x = std::max(x, 1e-30);
			Float dxnedd   = ye*x;

			// derivative with respect to temperature
			Float dxnedt   = helmholtz_constants::h3(dsi0t,  dsi1t,  dsi0mt,  dsi1mt,
            	si0d,   si1d,   si0md,   si1md);

			// derivative with respect to abar and zbar
			Float dxneda = -x*din*ytot1;
			Float dxnedz =  x *rho*ytot1;


			// the desired electron-positron thermodynamic quantities

			// dpepdd at high temperatures and low rhosities is below the
			// floating point limit of the subtraction of two large terms.
			// since dpresdd doesn't enter the maxwell relations at all, use the
			// bicubic interpolation done above instead of the formally correct expression
			x       = din*din;
			Float pele    = x*df_d;
			Float dpepdt  = x*df_dt;
			// dpepdd  = ye*(x*df_dd + 2.0*din*df_d)
			s       = dpepdd/ye - 2.0*din*df_d;
			Float dpepda  = -ytot1*(2.0*pele + s*din);
			Float dpepdz  = rho*ytot1*(2.0*din*df_d  +  s);


			x       = ye*ye;
			Float sele    = -df_t*ye;
			Float dsepdt  = -df_tt*ye;
			Float dsepdd  = -df_dt*x;
			Float dsepda  = ytot1*(ye*df_dt*din - sele);
			Float dsepdz  = -ytot1*(ye*df_dt*rho  + df_t);


			/* debug: */
			if (debug) std::cout << "dsepdt=" << dsepdt << " = -df_tt=" << df_tt << " * ye=" << ye << "\n";


			Float eele    = ye*free + T*sele;
			Float deepdt  = T*dsepdt;
			Float deepdd  = x*df_d + T*dsepdd;
			Float deepda  = -ye*ytot1*(free +  df_d*din) + T*dsepda;
			Float deepdz  = ytot1* (free + ye*df_d*rho) + T*dsepdz;


			/* debug: */
			if (debug) std::cout << "deepdt=" << deepdt << " = dsepdt=" << dsepdt << " * T" << "\n";


			// coulomb section:

			// uniform background corrections only
			// from yakovlev & shalybkov 1989
			// lami is the average ion seperation
			// plasg is the plasma coupling parameter

			z        = std::numbers::pi/4.;
			s        = z*xni;
			Float dsdd     = z*dxnidd;
			Float dsda     = z*dxnida;

			Float lami     = std::pow(1./s, 1./3.);
			Float inv_lami = 1./lami;
			z        = -lami/3;
			Float lamidd   = z*dsdd/s;
			Float lamida   = z*dsda/s;

			Float plasg    = zbar*zbar*helmholtz_constants::esqu*ktinv*inv_lami;
			z        = -plasg*inv_lami;
			Float plasgdd  = z*lamidd;
			Float plasgda  = z*lamida;
			Float plasgdt  = -plasg*ktinv*helmholtz_constants::kerg;
			Float plasgdz  = 2.0*plasg/zbar;

			Float ecoul, pcoul, scoul,
				decouldd, decouldt, decoulda, decouldz,
				dpcouldd, dpcouldt, dpcoulda, dpcouldz,
				dscouldd, dscouldt, dscoulda, dscouldz;

			// yakovlev & shalybkov 1989 equations 82, 85, 86, 87
			if (plasg >= 1.) {
				x        = std::pow(plasg, 0.25);
				y        = helmholtz_constants::avo*ytot1*helmholtz_constants::kerg;
				ecoul    = y*T*(helmholtz_constants::a1*plasg + helmholtz_constants::b1*x + helmholtz_constants::c1/x + helmholtz_constants::d1);
				pcoul    = rho*ecoul/3.;
				scoul    = -y*(3.0*helmholtz_constants::b1*x - 5.0*helmholtz_constants::c1/x + helmholtz_constants::d1*(std::log(plasg) - 1.) - helmholtz_constants::e1);

				y        = helmholtz_constants::avo*ytot1*kt*(helmholtz_constants::a1 + 0.25/plasg*(helmholtz_constants::b1*x - helmholtz_constants::c1/x));
				decouldd = y*plasgdd;
				decouldt = y*plasgdt + ecoul/T;
				decoulda = y*plasgda - ecoul/abar;
				decouldz = y*plasgdz;


				/* debug: */
				if (debug) std::cout << "decouldt=" << decouldt << " = y=" << y << " * plasgdt=" << decouldt << " + ecoul=" << ecoul << " / T" << "\n";


				y        = rho/3.;
				dpcouldd = ecoul + y*decouldd/3.;
				dpcouldt = y*decouldt;
				dpcoulda = y*decoulda;
				dpcouldz = y*decouldz;


				y        = -helmholtz_constants::avo*helmholtz_constants::kerg/(abar*plasg)*(0.75*helmholtz_constants::b1*x + 1.25*helmholtz_constants::c1/x + helmholtz_constants::d1);
				dscouldd = y*plasgdd;
				dscouldt = y*plasgdt;
				dscoulda = y*plasgda - scoul/abar;
				dscouldz = y*plasgdz;

			//yakovlev & shalybkov 1989 equations 102, 103, 104
			} else if (plasg < 1.) {
				x        = plasg*std::sqrt(plasg);
				y        = std::pow(plasg, helmholtz_constants::b2);
				z        = helmholtz_constants::c2*x - helmholtz_constants::a2*y/3.;
				pcoul    = -pion*z;
				ecoul    = 3.0*pcoul/rho;
				scoul    = -helmholtz_constants::avo/abar*helmholtz_constants::kerg*(helmholtz_constants::c2*x - helmholtz_constants::a2*(helmholtz_constants::b2 - 1.)/helmholtz_constants::b2*y);

				s        = 1.5*helmholtz_constants::c2*x/plasg - helmholtz_constants::a2*helmholtz_constants::b2*y/plasg/3.;
				dpcouldd = -dpiondd*z - pion*s*plasgdd;
				dpcouldt = -dpiondt*z - pion*s*plasgdt;
				dpcoulda = -dpionda*z - pion*s*plasgda;
				dpcouldz = -dpiondz*z - pion*s*plasgdz;

				s        = 3.0/rho;
				decouldd = s*dpcouldd - ecoul/rho;
				decouldt = s*dpcouldt;
				decoulda = s*dpcoulda;
				decouldz = s*dpcouldz;


				/* debug: */
				if (debug) std::cout << "decouldt=" << decouldt << " = s=" << s << " * dpcouldt=" << dpcouldt <<"\n";


				s        = -helmholtz_constants::avo*helmholtz_constants::kerg/(abar*plasg)*(1.5*helmholtz_constants::c2*x - helmholtz_constants::a2*(helmholtz_constants::b2 - 1.)*y);
				dscouldd = s*plasgdd;
				dscouldt = s*plasgdt;
				dscoulda = s*plasgda - scoul/abar;
				dscouldz = s*plasgdz;
			}





			// bomb proof
			x   = prad + pion + pele + pcoul;
			y   = erad + eion + eele + ecoul;
			z   = srad + sion + sele + scoul;

			// if (x .le. 0.0 .or. y .le. 0.0 .or. z .le. 0.0) then
			// if (x .le. 0.0) then
			if (x <= 0. || y <= 0.) {
				pcoul    = 0.;
				dpcouldd = 0.;
				dpcouldt = 0.;
				dpcoulda = 0.;
				dpcouldz = 0.;
				ecoul    = 0.;
				decouldd = 0.;
				decouldt = 0.;
				decoulda = 0.;
				decouldz = 0.;
				scoul    = 0.;
				dscouldd = 0.;
				dscouldt = 0.;
				dscoulda = 0.;
				dscouldz = 0.;
			}


			// sum all the gas components
			Float pgas    = pion + pele + pcoul;
			Float egas    = eion + eele + ecoul;
			Float sgas    = sion + sele + scoul;

			Float dpgasdd = dpiondd + dpepdd + dpcouldd;
			Float dpgasdt = dpiondt + dpepdt + dpcouldt;
			Float dpgasda = dpionda + dpepda + dpcoulda;
			Float dpgasdz = dpiondz + dpepdz + dpcouldz;

			Float degasdd = deiondd + deepdd + decouldd;
			Float degasdt = deiondt + deepdt + decouldt;
			Float degasda = deionda + deepda + decoulda;
			Float degasdz = deiondz + deepdz + decouldz;

			Float dsgasdd = dsiondd + dsepdd + dscouldd;
			Float dsgasdt = dsiondt + dsepdt + dscouldt;
			Float dsgasda = dsionda + dsepda + dscoulda;
			Float dsgasdz = dsiondz + dsepdz + dscouldz;


			/* debug: */
			if (debug) std::cout << "degasdt=" << degasdt << " = deiondt=" << deiondt << " + deepdt=" << deepdt << " + decouldt=" << decouldt << "\n";


			// add in radiation to get the total
			Float pres    = prad + pgas;
			Float ener    = erad + egas;
			Float entr    = srad + sgas;

			Float dpresdd = dpraddd + dpgasdd;
			Float dpresdt = dpraddt + dpgasdt;
			Float dpresda = dpradda + dpgasda;
			Float dpresdz = dpraddz + dpgasdz;

			Float rhoerdd = deraddd + degasdd;
			Float rhoerdt = deraddt + degasdt;
			Float rhoerda = deradda + degasda;
			Float rhoerdz = deraddz + degasdz;

			Float rhotrdd = dsraddd + dsgasdd;
			Float rhotrdt = dsraddt + dsgasdt;
			Float rhotrda = dsradda + dsgasda;
			Float rhotrdz = dsraddz + dsgasdz;


			/* debug: */
			if (debug) std::cout << "rhoerdt(cv)=" << rhoerdt << " = deraddt=" << deraddt << " + degasdt=" << degasdt << "\n\n";


			// for the gas
			// the temperature and rhosity exponents (c&g 9.81 9.82)
			// the specific heat at constant volume (c&g 9.92)
			// the third adiabatic exponent (c&g 9.93)
			// the first adiabatic exponent (c&g 9.97)
			// the second adiabatic exponent (c&g 9.105)
			// the specific heat at constant pressure (c&g 9.98)
			// and relativistic formula for the sound speed (c&g 14.29)

			struct eos_output {
				Float cv, dP_dT, P;
				Float cp, sound;

				Float dse, dpe, dsp;
				Float cv_gaz, cp_gaz, sound_gaz; 
			} res;


			Float zz            = pgas*rhoi;
			Float zzi           = rho/pgas;
			Float chit_gas      = T/pgas*dpgasdt;
			Float chid_gas      = dpgasdd*zzi;
			res.cv_gaz    = degasdt;
			x             = zz*chit_gas/(T*res.cv_gaz);
			Float gam3_gas      = x + 1.;
			Float gam1_gas      = chit_gas*x + chid_gas;
			Float nabad_gas     = x/gam1_gas;
			Float gam2_gas      = 1./(1. - nabad_gas);
			res.cp_gaz    = res.cv_gaz*gam1_gas/chid_gas;
			z             = 1. + (egas + helmholtz_constants::clight*helmholtz_constants::clight)*zzi;
			res.sound_gaz = helmholtz_constants::clight*std::sqrt(gam1_gas/z);



			// for the totals
			zz    = pres*rhoi;
			zzi   = rho/pres;
			Float chit  = T/pres*dpresdt;
			Float chid  = dpresdd*zzi;
			res.cv    = rhoerdt;
			x     = zz*chit/(T*res.cv);
			Float gam3  = x + 1.;
			Float gam1  = chit*x + chid;
			Float nabad = x/gam1;
			Float gam2  = 1./(1. - nabad);
			res.cp    = res.cv*gam1/chid;
			z     = 1. + (ener + helmholtz_constants::clight*helmholtz_constants::clight)*zzi;
			res.sound = helmholtz_constants::clight*std::sqrt(gam1/z);



			// maxwell relations; each is zero if the consistency is perfect
			x   = rho*rho;
			res.dse = T*rhotrdt/rhoerdt - 1.;
			res.dpe = (rhoerdd*x + T*dpresdt)/pres - 1.;
			res.dsp = -rhotrdd*x/dpresdt - 1.;

			// Needed output
			res.dP_dT /*dUdYe*/=degasdz*abar;

			return res;
		}
	};
}

