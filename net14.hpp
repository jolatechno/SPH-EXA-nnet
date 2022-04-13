#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "net14-constants.hpp"

namespace nnet {
	namespace net14 {
		/// constant mass-excendent values
		const Eigen::VectorXd BE = [](){
				Eigen::VectorXd BE_(14);
				BE_ << 0.0, 7.27440, 14.43580, 19.16680, 28.48280, 38.46680, 45.41480, 52.05380, 59.09380, 64.22080, 71.91280, 79.85180, 87.84680, 90.55480;
				return BE_*constants::UNKNOWN;
			}();

		/// constant number of particle created by photodesintegrations
		const Eigen::Matrix<int, -1, -1> n_photodesintegration = [](){
				Eigen::Matrix<int, -1, -1> n = Eigen::Matrix<int, -1, -1>::Zero(14, 14);

				// C -> 3He
				n(0, 1) = 3;

				// Z + He <- Z "+ 1"
				for (int i = 1; i < 13; ++i) {
					n(i, i + 1) = 2;
					n(0, i + 1) = 2;
				}

				return n;
			}();

		/// constant number of particle created by fusions
		const Eigen::Tensor<float, 3> n_fusion = [](){
				Eigen::Tensor<float, 3> n(14, 14, 14);
				n.setZero();

				// Z + He -> Z "+ 1"
				for (int i = 1; i < 13; ++i)
					n(i + 1, 0, i) = 1;

				// C + C -> Ne + He
				n(3, 1, 1) = 2;
				n(0, 1, 1) = 2;

				// C + O -> Mg + He
				n(4, 1, 2) = 2;
				n(0, 1, 2) = 2;

				// O + O -> Si + He
				n(5, 2, 2) = 2;
				n(0, 2, 2) = 2;

				// 3He -> C
				n(1, 0, 0) = 2./3.;

				return n;
			}();

		/// function computing the photodesintegration rates withing net 14
		template<typename Float>
		Eigen::Matrix<Float, -1, -1> get_photodesintegration_rates(const Float T) {
			/* -------------------
			simply copute desintegration rates within net-14
			------------------- */

			Eigen::Matrix<Float, -1, -1> r = Eigen::Matrix<Float, -1, -1>::Zero(14, 14);

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			bellow is implemented the weigth sof the following reactions, based on a paper using fits:
				- Ne + He <- Mg
				- Mg + He <- Si
				- Si + He <- S
				-  S + He <- Ar
				- Ar + He <- Ca
				- Ca + He <- Ti
				- Ti + He <- Cr
				- Cr + He <- Fe
				- Fe + He <- Ni
				- Ni + He <- Zn
			!!!!!!!!!!!!!!!!!!!!!!!! */

			Float t9=T/1.e9;
			Float t913=std::pow(t9, 1./3.);
			Float t923=t913*t913;
			Float t953=std::pow(t9, 5./3.);
			Float t9i=1.e0/t9;
			Float t9i2=t9i*t9i;
			Float t9i13=1.e0/t913;
			Float t9i23=t9i13*t9i13;
			Float t9i43=t9i23*t9i23;
			Float lt9=std::log(t9);

			Float val1=11.6045e0*t9i;
			Float val2=1.5e0*lt9;
			Float val3=val1*t9i*1.e-9;
			Float val4=1.5e-9*t9i;

			// Z + He <- Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				Float coef = 
					  constants::fits::fit[i - 4][1]*t9i
					+ constants::fits::fit[i - 4][2]*t9i13
					+ constants::fits::fit[i - 4][3]*t913
					+ constants::fits::fit[i - 4][4]*t9
					+ constants::fits::fit[i - 4][5]*t953
					+ constants::fits::fit[i - 4][6]*lt9;

				int k = constants::fits::get_temperature_range(T);
				Float rate = constants::fits::choose[i - 4][k]/constants::fits::choose[i + 1 - 4][k]*
					std::exp(
						  coef
						+ constants::fits::fit[i - 4][7]
						- constants::fits::q  [i - 4]*val1
						+ val2
					);
				r(i - 1, i) = rate;
				r(0, i) = rate;
			}


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C -> 3He
			!!!!!!!!!!!!!!!!!!!!!!!! */

      		Float t9r=T*1.0e-09;
      		t9=std::min((Float)10., t9r);
      		Float t92=t9*t9;
      		Float t93=t92*t9;
      		Float t95=t92*t93;
      		Float t912=std::sqrt(t9);
      		t913=std::pow(t9, 1./3.);
      		t923=t913*t913;
      		Float t932=t9*t912;
      		Float t943=t9*t913;
      		t953=t9*t923;

      		t9i=1./t9;
      		t9i13=1./t913;
      	 	t9i23=1./t923;
      		Float t9i32=1./t932;
      		Float t9rm1=1./t9r;
      		Float t9r32=t9*std::sqrt(t9r);

			Float r2abe = (7.40e+05*t9i32)*std::exp(-1.0663*t9i)
				+ 4.164e+09*t9i23*std::exp(-13.49*t9i13-t92/0.009604)*(1. + 0.031*t913 + 8.009*t923 + 1.732*t9 + 49.883*t943 + 27.426*t953);
      		Float r3a, rbeac = (130.*t9i32)*std::exp(-3.3364*t9i)
      			+ 2.510e+07*t9i23*std::exp(-23.57*t9i13 - t92/0.055225)*(1. + 0.018*t913 + 5.249*t923 + 0.650*t9 + 19.176*t943 + 6.034*t953);
			if(T > 8e7) {
	      		r3a=2.90e-16*(r2abe*rbeac)
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
		    } else
	      		r3a=2.90e-16*(r2abe*rbeac)*(0.01 + 0.2*(1. + 4.*std::exp(-std::pow(0.025*t9i, 3.263)))/(1. + 4.*std::exp(-std::pow(t9/0.025, 9.227))))
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
			Float rev = 2.e20*std::exp(-84.419412e0*t9i);

			r(0, 1) = rev*(t93)*r3a;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C + He <- O
			!!!!!!!!!!!!!!!!!!!!!!!! */


			Float rcag = 1.04e+08/(t92*std::pow(1. + 0.0489*t9i23, 2.))*std::exp(-32.120*t9i13-t92/12.222016)
            	+ 1.76e+08/(t92*std::pow(1. + 0.2654*t9i23, 2.))*std::exp(-32.120*t9i13)
           			+ 1.25e+03*t9i32*std::exp(-27.499*t9i)
           			+ 1.43e-02*t95*std::exp(-15.541*t9i);
			Float roga = rcag*5.13e+10*t9r32*std::exp(-83.108047*t9rm1);
			r(1, 2) = roga;
			r(0, 2) = roga;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				O + He <- Ne
			!!!!!!!!!!!!!!!!!!!!!!!! */


			Float roag=(9.37e+09*t9i23*std::exp(-39.757*t9i13-t92/2.515396) + 62.1*t9i32*std::exp(-10.297*t9i) + 538.*t9i32*std::exp(-12.226*t9i) + 13.*t92*std::exp(-20.093*t9i));
      		Float rnega=roag*5.65e+10*t9r32*std::exp(-54.93807*t9rm1);
      		r(2, 3)=rnega;
      		r(0, 3)=rnega;


			return r;
		}

		/// function computing the fusions rates withing net 14
		template<typename Float>
		Eigen::Tensor<Float, 3> get_fusion_rates(const Float T) {
			/* -------------------
			simply copute fusion rates within net-14
			------------------- */

			Eigen::Tensor<Float, 3> f(14, 14, 14);
			f.setZero();

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			bellow is implemented the weigth sof the following reactions, based on a paper using fits:
				- Ne + He -> Mg
				- Mg + He -> Si
				- Si + He -> S
				-  S + He -> Ar
				- Ar + He -> Ca
				- Ca + He -> Ti
				- Ti + He -> Cr
				- Cr + He -> Fe
				- Fe + He -> Ni
				- Ni + He -> Zn
			!!!!!!!!!!!!!!!!!!!!!!!! */

			{
				Float t9=T/1.e9;
				Float t913=std::pow(t9, 1./3.);
				Float t923=t913*t913;
				Float t953=std::pow(t9, 5./3.);
				Float t9i=1.e0/t9;
				Float t9i2=t9i*t9i;
				Float t9i13=1.e0/t913;
				Float t9i23=t9i13*t9i13;
				Float t9i43=t9i23*t9i23;
				Float lt9=std::log(t9);

				// Z + He -> Z "+ 1"
				for (int i = 3; i < 14; ++i) {
					Float coef = 
						  constants::fits::fit[i - 4][1]*t9i
						+ constants::fits::fit[i - 4][2]*t9i13
						+ constants::fits::fit[i - 4][3]*t913
						+ constants::fits::fit[i - 4][4]*t9
						+ constants::fits::fit[i - 4][5]*t953
						+ constants::fits::fit[i - 4][6]*lt9;
					f(i, 0, i - 1) = std::exp(constants::fits::fit[i - 4][0] + coef); 
				}
			}


      		/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				3He -> C
			!!!!!!!!!!!!!!!!!!!!!!!! */

			Float t9r=T*1.0e-09;
      		Float t9=std::min((Float)10., t9r);
      		Float t92=t9*t9;
      		Float t93=t92*t9;
      		Float t95=t92*t93;
      		Float t912=std::sqrt(t9);
      		Float t913=std::pow(t9, 1./3.);
      		Float t923=t913*t913;
      		Float t932=t9*t912;
      		Float t943=t9*t913;
      		Float t953=t9*t923;

      		Float t9i=1./t9;
      		Float t9i13=1./t913;
      	 	Float t9i23=1./t923;
      		Float t9i32=1./t932;
      		Float t9rm1=1./t9r;
      		Float t9r32=t9*std::sqrt(t9r);

      		Float t9a=t9/(1. + 0.0396*t9);
		    Float t9a13=std::pow(t9a, 1./3.);
		    Float t9a56=std::pow(t9a, 5./6.);


			Float r2abe = (7.40e+05*t9i32)*std::exp(-1.0663*t9i)
				+ 4.164e+09*t9i23*std::exp(-13.49*t9i13-t92/0.009604)*(1. + 0.031*t913 + 8.009*t923 + 1.732*t9 + 49.883*t943 + 27.426*t953);
      		Float r3a, rbeac = (130.*t9i32)*std::exp(-3.3364*t9i)
      			+ 2.510e+07*t9i23*std::exp(-23.57*t9i13 - t92/0.055225)*(1. + 0.018*t913 + 5.249*t923 + 0.650*t9 + 19.176*t943 + 6.034*t953);
			if(T > 8e7) {
	      		r3a=2.90e-16*(r2abe*rbeac)
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
		    } else
	      		r3a=2.90e-16*(r2abe*rbeac)*(0.01 + 0.2*(1. + 4.*std::exp(-std::pow(0.025*t9i, 3.263)))/(1. + 4.*std::exp(-std::pow(t9/0.025, 9.227))))
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);	
	      	f(1, 0, 0) = r3a;


	      	/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				2C -> Ne + He
			!!!!!!!!!!!!!!!!!!!!!!!! */
      		Float r24=4.27e+26*t9a56*t9i32*std::exp(-84.165/t9a13 - 2.12e-03*t93);
      		f(3, 1, 1) = r24;
      		f(0, 1, 1) = r24;


      		/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C + He -> O
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// C -> O
			Float rcag = 1.04e+08/(t92*std::pow(1. + 0.0489*t9i23, 2.))*std::exp(-32.120*t9i13-t92/12.222016)
            	+ 1.76e+08/(t92*std::pow(1. + 0.2654*t9i23, 2.))*std::exp(-32.120*t9i13)
           			+ 1.25e+03*t9i32*std::exp(-27.499*t9i)
           			+ 1.43e-02*t95*std::exp(-15.541*t9i);
			f(2, 0, 1) = rcag;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				O + He -> Ne
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// O -> Ne
			Float roag=(9.37e+09*t9i23*std::exp(-39.757*t9i13-t92/2.515396) + 62.1*t9i32*std::exp(-10.297*t9i) + 538.*t9i32*std::exp(-12.226*t9i) + 13.*t92*std::exp(-20.093*t9i));
        	f(3, 0, 2)=roag;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C + O -> Mg + He
			!!!!!!!!!!!!!!!!!!!!!!!! */
			Float r1216=0.;
			if (T > 5e9) {
	            t9a=t9/(1. + 0.055*t9);
	            Float t9a2=t9a*t9a;
	            t9a13=std::pow(t9a, 1./3.);
	            Float t9a23=t9a13*t9a13;
	            t9a56=std::pow(t9a, 5./6.);
	            r1216=1.72e+31*t9a56*t9i32*std::exp(-106.594/t9a13)/(std::exp(-0.18*t9a2) + 1.06e-03*std::exp(2.562*t9a23));
	        }
	        f(4, 1, 2) = r1216;
      		f(0, 1, 2) = r1216;


      		/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				2O -> Si + He
			!!!!!!!!!!!!!!!!!!!!!!!! */
			Float r32=7.10d+36*t9i23*std::exp(-135.93*t9i13 - 0.629*t923 - 0.445*t943 + 0.0103*t92);
      		f(5, 2, 2) = r32;
      		f(0, 2, 2) = r32;

			return f;
		}

		/// function computing the ideal gaz correction
		template<typename Float>
		Eigen::Vector<Float, -1> ideal_gaz_correction(const Float T) {
			/* -------------------
			simply copute the coulombian correction of BE within net-14
			------------------- */

			Eigen::Vector<Float, -1> BE_corr(14);
			BE_corr = 3./2. * constants::Kb * constants::Na * T;

			return BE_corr;
		}

		/// function computing the coulombian correction
		template<class vector, typename Float>
		vector coulomb_correction(const vector &Y, const Float T, const Float rho) {
			/* -------------------
			simply copute the coulombian correction of BE within net-14
			------------------- */

			vector BE_corr(14);

			// TO VERIFY !!!!!

			BE_corr(0) = 0;
			for (int i = 1; i < 14; ++i) {
				BE_corr(i) = 2.27e5 * std::pow((Float)constants::Z(i), 5./3.) * std::pow(rho * Y(i) /* !!!!! verify ????? */, 1./3.) / T;
			}

			return BE_corr;
		}
	}
}