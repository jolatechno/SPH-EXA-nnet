#pragma once

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

#include "../CUDA/cuda.inl"

#include "../eigen/eigen.hpp"

#ifndef IMAX
	#define IMAX 541
#endif
#ifndef JMAX
	#define JMAX 201
#endif
#ifndef HELM_TABLE_PATH
	#define HELM_TABLE_PATH "./helm_table.dat"
#endif

#include <iostream>

#include <sstream>
#include <string>
#include <array>

#include <vector>
#include <tuple>
#include <math.h>

#ifdef USE_CUDA
	#include "cuda_runtime.h"
#endif

namespace nnet::eos {
	/* !!!!!!!!!!!!
	debuging :
	!!!!!!!!!!!! */
	bool debug = false;


	namespace helmholtz_constants {
		// table size
		static const int imax = IMAX, jmax = JMAX;

		// table type
		typedef std::array<double, imax> ivector; // double[imax]
		typedef std::array<double, jmax> jvector; // double[jmax]
		typedef std::array<double, imax - 1> imvector; // double[imax]
		typedef std::array<double, jmax - 1> jmvector; // double[jmax]
		typedef eigen::fixed_size_matrix<double, imax, jmax> ijmatrix; // double[imax][jmax]

		// table limits
		static const double tlo   = 3.;
		static const double thi   = 13.;
		static const double tstp  = (thi - tlo)/(double)(jmax - 1);
		static const double tstpi = 1./tstp;
		static const double dlo   = -12.;
		static const double dhi   = 15.;
		static const double dstp  = (dhi - dlo)/(double)(imax - 1);
		static const double dstpi = 1./dstp;

		// physical constants
		static const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319;
		static const double g       = 6.6742867e-8;
        static const double h       = 6.6260689633e-27;
        static const double hbar    = 0.5*h/pi;
        static const double qe      = 4.8032042712e-10;
        static const double avo     = 6.0221417930e23;
        static const double clight  = 2.99792458e10;
        static const double kerg    = 1.380650424e-16;
        static const double ev2erg  = 1.60217648740e-12;
        static const double kev     = kerg/ev2erg;
        static const double amu     = 1.66053878283e-24;
        static const double mn      = 1.67492721184e-24;
        static const double mp      = 1.67262163783e-24;
        static const double me      = 9.1093821545e-28;
        static const double rbohr   = hbar*hbar/(me*qe*qe);
        static const double fine    = qe*qe/(hbar*clight);
        static const double hion    = 13.605698140;
        static const double ssol    = 5.6704e-5;
        static const double asol    = 4.0*ssol / clight;
        static const double weinlam = h*clight/(kerg*4.965114232);
        static const double weinfre = 2.821439372*kerg/h;
        static const double rhonuc  = 2.342e14;
        static const double kergavo = kerg*avo;
		static const double sioncon = (2.0*pi*amu*kerg)/(h*h);

		// parameters
		static const double a1    = -0.898004;
        static const double b1    =  0.96786;
        static const double c1    =  0.220703;
        static const double d1    = -0.86097;
        static const double e1    =  2.5269;
        static const double a2    =  0.29561;
        static const double b2    =  1.9885;
        static const double c2    =  0.288675;
        static const double esqu  =  qe*qe;

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
			> read_table()
		{
			// read table
			const std::string helmolt_table = { 
				#include HELM_TABLE_PATH
			};

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
				double dsav = dlo + i*dstp;
				d[i] = std::pow(10., dsav);
			}
			for (int j = 0; j < jmax; ++j) {
				double tsav = tlo + j*tstp;
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
		auto const [
			d_, dd_sav_, dd2_sav_, ddi_sav_, dd2i_sav_, dd3i_sav_,
			t_, dt_sav_, dt2_sav_, dti_sav_, dt2i_sav_, dt3i_sav_,
			
			f_,
   			fd_, ft_,
   			fdd_, ftt_, fdt_,
   			fddt_, fdtt_, fddtt_,

   			dpdf_, dpdfd_, dpdft_, dpdfdt_,

   			ef_, efd_, eft_, efdt_,
   			xf_, xfd_, xft_, xfdt_
	   	] = read_table();


		CUDA_DEFINE(static const inline ivector,  d,        = d_;)
		CUDA_DEFINE(static const inline imvector, dd_sav,   = dd_sav_;)
		CUDA_DEFINE(static const inline imvector, dd2_sav,  = dd2_sav_;)
		CUDA_DEFINE(static const inline imvector, ddi_sav,  = ddi_sav_;)
		CUDA_DEFINE(static const inline imvector, dd2i_sav, = dd2i_sav_;)
		CUDA_DEFINE(static const inline imvector, dd3i_sav, = dd3i_sav_;)

		CUDA_DEFINE(static const inline jvector,  t,        = t_;)
		CUDA_DEFINE(static const inline jmvector, dt_sav,   = dt_sav_;)
		CUDA_DEFINE(static const inline jmvector, dt2_sav,  = dt2_sav_;)
		CUDA_DEFINE(static const inline jmvector, dti_sav,  = dti_sav_;)
		CUDA_DEFINE(static const inline jmvector, dt2i_sav, = dt2i_sav_;)
		CUDA_DEFINE(static const inline jmvector, dt3i_sav, = dt3i_sav_;)

		CUDA_DEFINE(static const inline ijmatrix, f,        = f_;)
		CUDA_DEFINE(static const inline ijmatrix, fd,       = fd_;)
		CUDA_DEFINE(static const inline ijmatrix, ft,       = ft_;)
		CUDA_DEFINE(static const inline ijmatrix, fdd,      = fdd_;)
		CUDA_DEFINE(static const inline ijmatrix, ftt,      = ftt_;)
		CUDA_DEFINE(static const inline ijmatrix, fdt,      = fdt_;)
		CUDA_DEFINE(static const inline ijmatrix, fddt,     = fddt_;)
		CUDA_DEFINE(static const inline ijmatrix, fdtt,     = fdtt_;)
		CUDA_DEFINE(static const inline ijmatrix, fddtt,    = fddtt_;)

		CUDA_DEFINE(static const inline ijmatrix, dpdf,     = dpdf_;)
		CUDA_DEFINE(static const inline ijmatrix, dpdfd,    = dpdfd_;)
		CUDA_DEFINE(static const inline ijmatrix, dpdft,    = dpdft_;)
		CUDA_DEFINE(static const inline ijmatrix, dpdfdt,   = dpdfdt_;)

		CUDA_DEFINE(static const inline ijmatrix, ef,       = ef_;)
		CUDA_DEFINE(static const inline ijmatrix, efd,      = efd_;)
		CUDA_DEFINE(static const inline ijmatrix, eft,      = eft_;)
		CUDA_DEFINE(static const inline ijmatrix, efdt,     = efdt_;)

		CUDA_DEFINE(static const inline ijmatrix, xf,       = xf_;)
		CUDA_DEFINE(static const inline ijmatrix, xfd,      = xfd_;)
		CUDA_DEFINE(static const inline ijmatrix, xft,      = xft_;)
		CUDA_DEFINE(static const inline ijmatrix, xfdt,     = xfdt_;)


		// quintic hermite polynomial statement functions
		// psi0 and its derivatives
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline psi0(const Float z) {
			return z*z*z*(z*(-6.*z + 15.) - 10.) + 1.;
		}
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline dpsi0(const Float z) {
			return z*z*(z*(-30.*z + 60.) - 30.);
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline  ddpsi0(const Float z) {
			return z*(z*(-120.*z + 180.) -60.);
		};

		// psi1 and its derivatives
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline psi1(const Float z) {
			return z*(z*z*(z*(-3.*z + 8.) - 6.) + 1.);
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline dpsi1(const Float z) {
			return z*z*(z*(-15.*z + 32.) - 18.) + 1.;
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline ddpsi1(const Float z) {
			return z*(z*(-60.*z + 96.) -36.);
		};

		// psi2  and its derivatives
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline psi2(const Float z) {
			return 0.5*z*z*(z*(z*(-z + 3.) - 3.) + 1.);
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline dpsi2(const Float z) {
			return  0.5*z*(z*(z*(-5.*z + 12.) - 9.) + 2.);
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline ddpsi2(const Float z) {
			return 0.5*(z*(z*(-20.*z + 36.) - 18.) + 2.);
		};

		// biquintic hermite polynomial statement function
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline h5(const Float *fi,
			const Float w0t, const Float w1t, const Float w2t, const Float w0mt, const Float w1mt, const Float w2mt,
			const Float w0d,const Float w1d, const Float w2d, const Float w0md, const Float w1md, const Float w2md)
		{
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
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline xpsi0(const Float z) {
			return z*z*(2.*z - 3.) + 1.;
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline xdpsi0(const Float z) {
			return z*(6.*z - 6.);
		};

		// psi1 & derivatives
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline xpsi1(const Float z) {
			return z*(z*(z - 2.) + 1.);
		};
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline xdpsi1(const Float z) {
			return z*(3.*z - 4.) + 1.;
		};

		// bicubic hermite polynomial statement function
		template<typename Float>
		CUDA_FUNCTION_DECORATOR Float inline h3(const Float *fi,
			const Float w0t, const Float w1t, const Float w0mt, const Float w1mt, const Float w0d, const Float w1d, const Float w0md, const Float w1md)
		{
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
		template<typename Float>
		CUDA_FUNCTION_DECORATOR std::pair<int, int> inline get_table_indices(const Float T, const Float rho, const Float abar, const Float zbar) {
			const Float ye = std::max(1e-16, zbar/abar);
			const Float din = ye*rho;

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
	CUDA_FUNCTION_DECORATOR auto inline helmholtz(double abar_, double zbar_, Float T, Float rho) {
		// coefs
		Float fi[36];

		Float abar = 1/abar_;
		Float zbar = zbar_/abar_;


		/* debug: */
		// if (debug) std::cout << "T=" << T << ", rho=" << rho << ", abar=" << abar << ", zbar=" << zbar << "\n";


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
		fi[0]  = CUDA_ACCESS(helmholtz_constants::f)(iat + 0, jat + 0);
		fi[1]  = CUDA_ACCESS(helmholtz_constants::f)(iat + 1, jat + 0);
		fi[2]  = CUDA_ACCESS(helmholtz_constants::f)(iat + 0, jat + 1);
		fi[3]  = CUDA_ACCESS(helmholtz_constants::f)(iat + 1, jat + 1);
		fi[4]  = CUDA_ACCESS(helmholtz_constants::ft)(iat + 0, jat + 0);
		fi[5]  = CUDA_ACCESS(helmholtz_constants::ft)(iat + 1, jat + 0);
		fi[6]  = CUDA_ACCESS(helmholtz_constants::ft)(iat + 0, jat + 1);
		fi[7]  = CUDA_ACCESS(helmholtz_constants::ft)(iat + 1, jat + 1);
		fi[8]  = CUDA_ACCESS(helmholtz_constants::ftt)(iat + 0, jat + 0);
		fi[9]  = CUDA_ACCESS(helmholtz_constants::ftt)(iat + 1, jat + 0);
		fi[10] = CUDA_ACCESS(helmholtz_constants::ftt)(iat + 0, jat + 1);
		fi[11] = CUDA_ACCESS(helmholtz_constants::ftt)(iat + 1, jat + 1);
		fi[12] = CUDA_ACCESS(helmholtz_constants::fd)(iat + 0, jat + 0);
		fi[13] = CUDA_ACCESS(helmholtz_constants::fd)(iat + 1, jat + 0);
		fi[14] = CUDA_ACCESS(helmholtz_constants::fd)(iat + 0, jat + 1);
		fi[15] = CUDA_ACCESS(helmholtz_constants::fd)(iat + 1, jat + 1);
		fi[16] = CUDA_ACCESS(helmholtz_constants::fdd)(iat + 0, jat + 0);
		fi[17] = CUDA_ACCESS(helmholtz_constants::fdd)(iat + 1, jat + 0);
		fi[18] = CUDA_ACCESS(helmholtz_constants::fdd)(iat + 0, jat + 1);
		fi[19] = CUDA_ACCESS(helmholtz_constants::fdd)(iat + 1, jat + 1);
		fi[20] = CUDA_ACCESS(helmholtz_constants::fdt)(iat + 0, jat + 0);
		fi[21] = CUDA_ACCESS(helmholtz_constants::fdt)(iat + 1, jat + 0);
		fi[22] = CUDA_ACCESS(helmholtz_constants::fdt)(iat + 0, jat + 1);
		fi[23] = CUDA_ACCESS(helmholtz_constants::fdt)(iat + 1, jat + 1);
		fi[24] = CUDA_ACCESS(helmholtz_constants::fddt)(iat + 0, jat + 0);
		fi[25] = CUDA_ACCESS(helmholtz_constants::fddt)(iat + 1, jat + 0);
		fi[26] = CUDA_ACCESS(helmholtz_constants::fddt)(iat + 0, jat + 1);
		fi[27] = CUDA_ACCESS(helmholtz_constants::fddt)(iat + 1, jat + 1);
		fi[28] = CUDA_ACCESS(helmholtz_constants::fdtt)(iat + 0, jat + 0);
		fi[29] = CUDA_ACCESS(helmholtz_constants::fdtt)(iat + 1, jat + 0);
		fi[30] = CUDA_ACCESS(helmholtz_constants::fdtt)(iat + 0, jat + 1);
		fi[31] = CUDA_ACCESS(helmholtz_constants::fdtt)(iat + 1, jat + 1);
		fi[32] = CUDA_ACCESS(helmholtz_constants::fddtt)(iat + 0, jat + 0);
		fi[33] = CUDA_ACCESS(helmholtz_constants::fddtt)(iat + 1, jat + 0);
		fi[34] = CUDA_ACCESS(helmholtz_constants::fddtt)(iat + 0, jat + 1);
		fi[35] = CUDA_ACCESS(helmholtz_constants::fddtt)(iat + 1, jat + 1);



		// various differences
		Float xt  = std::max( (T - CUDA_ACCESS(helmholtz_constants::t)[jat])*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat], 0.);
		Float xd  = std::max( (din - CUDA_ACCESS(helmholtz_constants::d)[iat])*CUDA_ACCESS(helmholtz_constants::ddi_sav)[iat], 0.);
		Float mxt = 1. - xt;
		Float mxd = 1. - xd;


		/* debug: */
		// if (debug) std::cout << "xt=" << xt << " = (T - t[" << jat << "]=" << CUDA_ACCESS(helmholtz_constants::t)[jat] << ")* dti_sav[" << jat << "]=" << CUDA_ACCESS(helmholtz_constants::dti_sav)[jat] << "\n";


		// the six rhosity and six temperature basis functions;
		Float si0t =   helmholtz_constants::psi0(xt);
		Float si1t =   helmholtz_constants::psi1(xt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];
		Float si2t =   helmholtz_constants::psi2(xt)*CUDA_ACCESS(helmholtz_constants::dt2_sav)[jat];


		/* debug: */
		// if (debug) std::cout << "si0t=" << si0t << " = psi0(xt=" << xt << ")\n";


		Float si0mt =  helmholtz_constants::psi0(mxt);
		Float si1mt = -helmholtz_constants::psi1(mxt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];
		Float si2mt =  helmholtz_constants::psi2(mxt)*CUDA_ACCESS(helmholtz_constants::dt2_sav)[jat];

		Float si0d =   helmholtz_constants::psi0(xd);
		Float si1d =   helmholtz_constants::psi1(xd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];
		Float si2d =   helmholtz_constants::psi2(xd)*CUDA_ACCESS(helmholtz_constants::dd2_sav)[iat];

		Float si0md =  helmholtz_constants::psi0(mxd);
		Float si1md = -helmholtz_constants::psi1(mxd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];
		Float si2md =  helmholtz_constants::psi2(mxd)*CUDA_ACCESS(helmholtz_constants::dd2_sav)[iat];

		// derivatives of the weight functions
		Float dsi0t =   helmholtz_constants::dpsi0(xt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		Float dsi1t =   helmholtz_constants::dpsi1(xt);
		Float dsi2t =   helmholtz_constants::dpsi2(xt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];

		Float dsi0mt = -helmholtz_constants::dpsi0(mxt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		Float dsi1mt =  helmholtz_constants::dpsi1(mxt);
		Float dsi2mt = -helmholtz_constants::dpsi2(mxt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];

		Float dsi0d =   helmholtz_constants::dpsi0(xd)*CUDA_ACCESS(helmholtz_constants::ddi_sav)[iat];
		Float dsi1d =   helmholtz_constants::dpsi1(xd);
		Float dsi2d =   helmholtz_constants::dpsi2(xd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];

		Float dsi0md = -helmholtz_constants::dpsi0(mxd)*CUDA_ACCESS(helmholtz_constants::ddi_sav)[iat];
		Float dsi1md =  helmholtz_constants::dpsi1(mxd);
		Float dsi2md = -helmholtz_constants::dpsi2(mxd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];

		// second derivatives of the weight functions
		Float ddsi0t =   helmholtz_constants::ddpsi0(xt)*CUDA_ACCESS(helmholtz_constants::dt2i_sav)[jat];
		Float ddsi1t =   helmholtz_constants::ddpsi1(xt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		Float ddsi2t =   helmholtz_constants::ddpsi2(xt);

		Float ddsi0mt =  helmholtz_constants::ddpsi0(mxt)*CUDA_ACCESS(helmholtz_constants::dt2i_sav)[jat];
		Float ddsi1mt = -helmholtz_constants::ddpsi1(mxt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		Float ddsi2mt =  helmholtz_constants::ddpsi2(mxt);

		// ddsi0d =   ddpsi0(xd)*dd2i_sav[iat];
		// ddsi1d =   ddpsi1(xd)*ddi_sav[iat];
		// ddsi2d =   ddpsi2(xd);

		// ddsi0md =  ddpsi0(mxd)*dd2i_sav[iat];
		// ddsi1md = -ddpsi1(mxd)*ddi_sav[iat];
		// ddsi2md =  ddpsi2(mxd);


		// the free energy
		Float free  = helmholtz_constants::h5(fi,
			si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
			si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

		// derivative with respect to rhosity
		Float df_d  = helmholtz_constants::h5(fi,
			si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
			dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);


		// derivative with respect to temperature
		Float df_t = helmholtz_constants::h5(fi,
			dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
			si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

		// derivative with respect to rhosity**2
		// df_dd = h5(fi,
		//		si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt,
		//		ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md, ddsi2md)

		// derivative with respect to temperature**2
		Float df_tt = helmholtz_constants::h5(fi,
			ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt,
			si0d,   si1d,   si2d,   si0md,   si1md,   si2md);



		// derivative with respect to temperature and rhosity
		Float df_dt = helmholtz_constants::h5(fi,
			dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
			dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);



		// now get the pressure derivative with rhosity, chemical potential, and
		// electron positron number rhosities
		// get the interpolation weight functions
		si0t   =  helmholtz_constants::xpsi0(xt);
		si1t   =  helmholtz_constants::xpsi1(xt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];

		si0mt  =  helmholtz_constants::xpsi0(mxt);
		si1mt  =  -helmholtz_constants::xpsi1(mxt)*CUDA_ACCESS(helmholtz_constants::dt_sav)[jat];

		si0d   =  helmholtz_constants::xpsi0(xd);
		si1d   =  helmholtz_constants::xpsi1(xd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];

		si0md  =  helmholtz_constants::xpsi0(mxd);
		si1md  =  -helmholtz_constants::xpsi1(mxd)*CUDA_ACCESS(helmholtz_constants::dd_sav)[iat];


		// derivatives of weight functions
		dsi0t  = helmholtz_constants::xdpsi0(xt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		dsi1t  = helmholtz_constants::xdpsi1(xt);

		dsi0mt = -helmholtz_constants::xdpsi0(mxt)*CUDA_ACCESS(helmholtz_constants::dti_sav)[jat];
		dsi1mt = helmholtz_constants::xdpsi1(mxt);

		dsi0d  = helmholtz_constants::xdpsi0(xd)*CUDA_ACCESS(helmholtz_constants::ddi_sav)[iat];
		dsi1d  = helmholtz_constants::xdpsi1(xd);

		dsi0md = -helmholtz_constants::xdpsi0(mxd)*CUDA_ACCESS(helmholtz_constants::ddi_sav)[iat];
		dsi1md = helmholtz_constants::xdpsi1(mxd);





		// move table values into coefficient table
		fi[0]  = CUDA_ACCESS(helmholtz_constants::dpdf)(iat + 0, jat + 0);
		fi[1]  = CUDA_ACCESS(helmholtz_constants::dpdf)(iat + 1, jat + 0);
		fi[2]  = CUDA_ACCESS(helmholtz_constants::dpdf)(iat + 0, jat + 1);
		fi[3]  = CUDA_ACCESS(helmholtz_constants::dpdf)(iat + 1, jat + 1);
		fi[4]  = CUDA_ACCESS(helmholtz_constants::dpdft)(iat + 0, jat + 0);
		fi[5]  = CUDA_ACCESS(helmholtz_constants::dpdft)(iat + 1, jat + 0);
		fi[6]  = CUDA_ACCESS(helmholtz_constants::dpdft)(iat + 0, jat + 1);
		fi[7]  = CUDA_ACCESS(helmholtz_constants::dpdft)(iat + 1, jat + 1);
		fi[8]  = CUDA_ACCESS(helmholtz_constants::dpdfd)(iat + 0, jat + 0);
		fi[9]  = CUDA_ACCESS(helmholtz_constants::dpdfd)(iat + 1, jat + 0);
		fi[10] = CUDA_ACCESS(helmholtz_constants::dpdfd)(iat + 0, jat + 1);
		fi[11] = CUDA_ACCESS(helmholtz_constants::dpdfd)(iat + 1, jat + 1);
		fi[12] = CUDA_ACCESS(helmholtz_constants::dpdfdt)(iat + 0, jat + 0);
		fi[13] = CUDA_ACCESS(helmholtz_constants::dpdfdt)(iat + 1, jat + 0);
		fi[14] = CUDA_ACCESS(helmholtz_constants::dpdfdt)(iat + 0, jat + 1);
		fi[15] = CUDA_ACCESS(helmholtz_constants::dpdfdt)(iat + 1, jat + 1);




		Float dpepdd  = helmholtz_constants::h3(fi,
			si0t,   si1t,   si0mt,   si1mt,
            si0d,   si1d,   si0md,   si1md);
			dpepdd  = std::max(ye*dpepdd, 1.e-30);





		// move table values into coefficient table
		fi[0]  = CUDA_ACCESS(helmholtz_constants::ef)(iat + 0, jat + 0);
		fi[1]  = CUDA_ACCESS(helmholtz_constants::ef)(iat + 1, jat + 0);
		fi[2]  = CUDA_ACCESS(helmholtz_constants::ef)(iat + 0, jat + 1);
		fi[3]  = CUDA_ACCESS(helmholtz_constants::ef)(iat + 1, jat + 1);
		fi[4]  = CUDA_ACCESS(helmholtz_constants::eft)(iat + 0, jat + 0);
		fi[5]  = CUDA_ACCESS(helmholtz_constants::eft)(iat + 1, jat + 0);
		fi[6]  = CUDA_ACCESS(helmholtz_constants::eft)(iat + 0, jat + 1);
		fi[7]  = CUDA_ACCESS(helmholtz_constants::eft)(iat + 1, jat + 1);
		fi[8]  = CUDA_ACCESS(helmholtz_constants::efd)(iat + 0, jat + 0);
		fi[9]  = CUDA_ACCESS(helmholtz_constants::efd)(iat + 1, jat + 0);
		fi[10] = CUDA_ACCESS(helmholtz_constants::efd)(iat + 0, jat + 1);
		fi[11] = CUDA_ACCESS(helmholtz_constants::efd)(iat + 1, jat + 1);
		fi[12] = CUDA_ACCESS(helmholtz_constants::efdt)(iat + 0, jat + 0);
		fi[13] = CUDA_ACCESS(helmholtz_constants::efdt)(iat + 1, jat + 0);
		fi[14] = CUDA_ACCESS(helmholtz_constants::efdt)(iat + 0, jat + 1);
		fi[15] = CUDA_ACCESS(helmholtz_constants::efdt)(iat + 1, jat + 1);






		// electron chemical potential etaele
		Float etaele  = helmholtz_constants::h3(fi,
			si0t,   si1t,   si0mt,   si1mt,
			si0d,   si1d,   si0md,   si1md);


		// derivative with respect to rhosity
		x = helmholtz_constants::h3(fi,
			si0t,   si1t,   si0mt,   si1mt,
			dsi0d,  dsi1d,  dsi0md,  dsi1md);
		Float detadd  = ye*x;

		// derivative with respect to temperature
		Float detadt  = helmholtz_constants::h3(fi,
			dsi0t,  dsi1t,  dsi0mt,  dsi1mt,
			si0d,   si1d,   si0md,   si1md);

		// derivative with respect to abar and zbar
		Float detada = -x*din*ytot1;
		Float detadz =  x*rho*ytot1;





		// move table values into coefficient table
		fi[0]  = CUDA_ACCESS(helmholtz_constants::xf)(iat + 0, jat + 0);
		fi[1]  = CUDA_ACCESS(helmholtz_constants::xf)(iat + 1, jat + 0);
		fi[2]  = CUDA_ACCESS(helmholtz_constants::xf)(iat + 0, jat + 1);
		fi[3]  = CUDA_ACCESS(helmholtz_constants::xf)(iat + 1, jat + 1);
		fi[4]  = CUDA_ACCESS(helmholtz_constants::xft)(iat + 0, jat + 0);
		fi[5]  = CUDA_ACCESS(helmholtz_constants::xft)(iat + 1, jat + 0);
		fi[6]  = CUDA_ACCESS(helmholtz_constants::xft)(iat + 0, jat + 1);
		fi[7]  = CUDA_ACCESS(helmholtz_constants::xft)(iat + 1, jat + 1);
		fi[8]  = CUDA_ACCESS(helmholtz_constants::xfd)(iat + 0, jat + 0);
		fi[9]  = CUDA_ACCESS(helmholtz_constants::xfd)(iat + 1, jat + 0);
		fi[10] = CUDA_ACCESS(helmholtz_constants::xfd)(iat + 0, jat + 1);
		fi[11] = CUDA_ACCESS(helmholtz_constants::xfd)(iat + 1, jat + 1);
		fi[12] = CUDA_ACCESS(helmholtz_constants::xfdt)(iat + 0, jat + 0);
		fi[13] = CUDA_ACCESS(helmholtz_constants::xfdt)(iat + 1, jat + 0);
		fi[14] = CUDA_ACCESS(helmholtz_constants::xfdt)(iat + 0, jat + 1);
		fi[15] = CUDA_ACCESS(helmholtz_constants::xfdt)(iat + 1, jat + 1);





		// electron + positron number rhosities
		Float xnefer = helmholtz_constants::h3(fi,
			si0t,   si1t,   si0mt,   si1mt,
        	si0d,   si1d,   si0md,   si1md);

		// derivative with respect to rhosity
		x = helmholtz_constants::h3(fi,
			si0t,   si1t,   si0mt,   si1mt,
        	dsi0d,  dsi1d,  dsi0md,  dsi1md);
		x = std::max(x, 1e-30);
		Float dxnedd   = ye*x;

		// derivative with respect to temperature
		Float dxnedt   = helmholtz_constants::h3(fi,
			dsi0t,  dsi1t,  dsi0mt,  dsi1mt,
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
		// if (debug) std::cout << "dsepdt=" << dsepdt << " = -df_tt=" << df_tt << " * ye=" << ye << "\n";


		Float eele    = ye*free + T*sele;
		Float deepdt  = T*dsepdt;
		Float deepdd  = x*df_d + T*dsepdd;
		Float deepda  = -ye*ytot1*(free +  df_d*din) + T*dsepda;
		Float deepdz  = ytot1* (free + ye*df_d*rho) + T*dsepdz;


		/* debug: */
		// if (debug) std::cout << "deepdt=" << deepdt << " = dsepdt=" << dsepdt << " * T" << "\n";


		// coulomb section:

		// uniform background corrections only
		// from yakovlev & shalybkov 1989
		// lami is the average ion seperation
		// plasg is the plasma coupling parameter

		z              = helmholtz_constants::pi*4./3.;
		s              = z*xni;
		Float dsdd     = z*dxnidd;
		Float dsda     = z*dxnida;

		/* debug: */
		// if (debug) std::cout << "s=" << s << " = z=" << z << " * xni=" << xni << "\n";


		Float lami     = std::pow(1./s, 1./3.);
		Float inv_lami = 1./lami;
		z              = -lami/3;
		Float lamidd   = z*dsdd/s;
		Float lamida   = z*dsda/s;

		Float plasg    = zbar*zbar*helmholtz_constants::esqu*ktinv*inv_lami;
		z        = -plasg*inv_lami;
		Float plasgdd  = z*lamidd;
		Float plasgda  = z*lamida;
		Float plasgdt  = -plasg*ktinv*helmholtz_constants::kerg;
		Float plasgdz  = 2.0*plasg/zbar;

		/* debug: */
		// if (debug) std::cout << "plasg=" << plasg << " = zbar=" << zbar << "^2 * esqu=" << helmholtz_constants::esqu << " * ktinv=" << ktinv << " * inv_lami=" << inv_lami << "\n";


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
			// if (debug) std::cout << "decouldt=" << decouldt << " = y=" << y << " * plasgdt=" << decouldt << " + ecoul=" << ecoul << " / T" << "\n";

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
			// if (debug) std::cout << "decouldt=" << decouldt << " = s=" << s << " * dpcouldt=" << dpcouldt <<"\n";


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
		// if (debug) std::cout << "degasdt=" << degasdt << " = deiondt=" << deiondt << " + deepdt=" << deepdt << " + decouldt=" << decouldt << "\n";


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
		// if (debug) std::cout << "rhoerdt(cv)=" << rhoerdt << " = deraddt=" << deraddt << " + degasdt=" << degasdt << "\n\n";


		// for the gas
		// the temperature and rhosity exponents (c&g 9.81 9.82)
		// the specific heat at constant volume (c&g 9.92)
		// the third adiabatic exponent (c&g 9.93)
		// the first adiabatic exponent (c&g 9.97)
		// the second adiabatic exponent (c&g 9.105)
		// the specific heat at constant pressure (c&g 9.98)
		// and relativistic formula for the sound speed (c&g 14.29)

		struct eos_output {
			Float cv, dP_dT, p;
			Float cp, c, u;

			Float dse, dpe, dsp;
			Float cv_gaz, cp_gaz, c_gaz; 

			Float dU_dYe;
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
		res.c_gaz = helmholtz_constants::clight*std::sqrt(gam1_gas/z);



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
		res.c = helmholtz_constants::clight*std::sqrt(gam1/z);



		// maxwell relations; each is zero if the consistency is perfect
		x   = rho*rho;
		res.dse = T*rhotrdt/rhoerdt - 1.;
		res.dpe = (rhoerdd*x + T*dpresdt)/pres - 1.;
		res.dsp = -rhotrdd*x/dpresdt - 1.;

		// Needed output
		res.dP_dT  = dpresdt;
		res.dU_dYe = degasdz*abar;
		res.p = pres;
		res.u = ener;

		return res;
	}




	/// helmholtz eos functor
	/**
	*...TODO
	 */
	template<typename Float=double>
	class helmholtz_functor {
	private:
		const Float *Z;
		int dimension;
#ifdef USE_CUDA
		Float *Z_dev;
#endif

		CUDA_FUNCTION_DECORATOR auto inline compute(const Float *Z_, const Float *Y, const Float T, const Float rho) const {
			// compute abar and zbar
			double abar = std::accumulate(Y, Y + dimension, (Float)0);
			double zbar = eigen::dot(Y, Y + dimension, Z);

			return helmholtz(abar, zbar, T, rho);
		}

	public:
		helmholtz_functor(const Float  *Z_, int dimension_) : Z(Z_), dimension(dimension_) {
#ifdef USE_CUDA
			cudaMalloc(&Z_dev, dimension*sizeof(Float));
			cudaMemcpy(Z_dev, Z, dimension*sizeof(Float), cudaMemcpyHostToDevice);
#endif
		}
		template<class Vector>
		helmholtz_functor(const Vector &Z_, int dimension_) : helmholtz_functor(Z_.data(), dimension_) {}
		template<class Vector>
		helmholtz_functor(const Vector &Z_) : helmholtz_functor(Z_.data(), Z_.size()) {}

		~helmholtz_functor() {
#ifdef USE_CUDA
			cudaFree(Z_dev);
#endif
		}

		CUDA_FUNCTION_DECORATOR auto inline operator()(const Float *Y, const Float T, const Float rho) const {
#ifdef  __CUDA_ARCH__
			return compute(Z_dev, Y, T, rho);
#else
			return compute(Z, Y, T, rho);
#endif
		}
	};
}

