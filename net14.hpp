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
				return BE_;
			}();

		/// constant number of particle created by photodesintegrations
		const Eigen::Matrix<int, 14, 14> n_photodesintegrations = [](){
				Eigen::Matrix<int, 14, 14> n = Eigen::Matrix<int, 14, 14>::Zero();

				// C -> 3He
				n(0, 1) = 3;

				// Z <-> Z "+ 1"
				for (int i = 1; i < 13; ++i) {
					n(i, i + 1) = 1;
					n(i + 1, i) = 1;
				}

				return n;
			}();

		/// constant number of particle created by fusions
		const Eigen::Tensor<int, 3> n_fusions = [](){
				Eigen::Tensor<int, 3> n(14, 14, 14);
				n.setZero();

				// C + C -> Ne + He
				n(3, 1, 1) = 2;
				n(0, 1, 1) = 2;

				// C + O -> Mg + He
				n(4, 1, 2) = 2;
				n(0, 1, 2) = 2;

				// O + O -> Si + He
				n(5, 2, 2) = 2;
				n(0, 2, 2) = 2;

				// 3He -> C ???? !!!!!
				
				return n;
			}();

		/// 
		template<typename Float>
		Eigen::Matrix<Float, 14, 14> get_photodesintegration_rates(const Float T) {
			/* -------------------
			simply copute desintegration rates within net-14
			------------------- */

			Eigen::Matrix<Float, 14, 14> r = Eigen::Matrix<Float, 14, 14>::Zero();

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			bellow is implemented the weigth sof the following reactions, based on a paper using fits:
				- Ne <-> Mg
				- Mg <-> Si
				- Si <-> S
				-  S <-> Ar
				- Ar <-> Ca
				- Ca <-> Ti
				- Ti <-> Cr
				- Cr <-> Fe
				- Fe <-> Ni
				- Ni <-> Zn
			!!!!!!!!!!!!!!!!!!!!!!!! */

			Float coefs[14 - 4] = {0.};

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

			// Z -> Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				coefs[i - 4] = 
					  constants::fits::fit[i - 4][1]
					+ constants::fits::fit[i - 4][2]*t9i
					+ constants::fits::fit[i - 4][3]*t9i13
					+ constants::fits::fit[i - 4][4]*t913
					+ constants::fits::fit[i - 4][5]*t9
					+ constants::fits::fit[i - 4][6]*t953
					+ constants::fits::fit[i - 4][7]*lt9;
				r(i, i - 1) = std::exp(coefs[i - 4]); 
			}

			Float val1=11.6045e0*t9i;
			Float val2=1.5e0*lt9;
			Float val3=val1*t9i*1.e-9;
			Float val4=1.5e-9*t9i;

			// Z <- Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				int k = constants::fits::get_temperature_range(T);
				r(i - 1, i) = constants::fits::choose[i - 4][k]/constants::fits::choose[i + 1 - 4][k]*
					std::exp(
						  constants::fits::fit[i - 4][8]
						+ coefs[i - 4]
						- constants::fits::q  [i - 4]*val1
						+ val2
					);
			}


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C -> He
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
      		Float rbeac = (130.*t9i32)*std::exp(-3.3364*t9i)
      			+ 2.510e+07*t9i23*std::exp(-23.57*t9i13 - t92/0.055225)*(1. + 0.018*t913 + 5.249*t923 + 0.650*t9 + 19.176*t943 + 6.034*t953);
			Float r3a, rev = 2.e20*std::exp(-84.419412e0*t9i);
			if(T > 8e7) {
	      		r3a=2.90e-16*(r2abe*rbeac)
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
		    } else
	      		r3a=2.90e-16*(r2abe*rbeac)*(0.01 + 0.2*(1. + 4.*std::exp(-std::pow(0.025*t9i, 3.263)))/(1. + 4.*std::exp(-std::pow(t9/0.025, 9.227))))
	      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);

			r(0, 1) = rev*(t93)*r3a;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C <-> O
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// C -> O
			Float rcag = 1.04e+08/(t92*std::pow(1. + 0.0489*t9i23, 2.))*std::exp(-32.120*t9i13-t92/12.222016)
            	+ 1.76e+08/(
            		t92*std::pow(1. + 0.2654*t9i23, 2.)*std::exp(-32.120*t9i13)
           			+ 1.25e+03*t9i32*std::exp(-27.499*t9i)
           			+ 1.43e-02*t95*std::exp(-15.541*t9i)
           		);
			r(2, 1) = rcag;

			// O -> C
			Float roga = rcag*5.13e+10*t9r32*std::exp(-83.108047*t9rm1);
			r(1, 2) = roga;


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				O <-> Ne
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// TODO !!!!!


			return r;
		}

		template<typename Float>
		Eigen::Tensor<Float, 3> get_fusion_rates(const Float T) {
			/* -------------------
			simply copute fusion rates within net-14
			------------------- */

			Eigen::Tensor<Float, 3> f(14, 14, 14);
			f.setZero();

			// C + C -> Ne + He
			// TODO !!!!!

			// C + O -> Mg + He
			// TODO !!!!!

			// O + O -> Si + He
			// TODO !!!!!
				
			// 3He -> C ???? !!!!!			

			return f;
		}

		/// function computing the coulombian correction
		template<typename Float>
		Eigen::Vector<Float, 14> ideal_gaz_correction(const Float T) {
			/* -------------------
			simply copute the coulombian correction of BE within net-14
			------------------- */

			Eigen::Vector<Float, 14> BE_corr(14);
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