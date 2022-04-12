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
		Eigen::Matrix<Float, 14, 14> get_net14_desintegration_rates(const Float T) {
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

			Float t9=temp/1.e9;
			Float t913=t9**(1.e0/3.e0);
			Float t923=t913*t913;
			Float t953=t9**(5.e0/3.e0);
			Float t9i=1.e0/t9;
			Float t9i2=t9i*t9i;
			Float t9i13=1.e0/t913;
			Float t9i23=t9i13*t9i13;
			Float t9i43=t9i23*t9i23;
			Float lt9=std::log(t9);

			// Z -> Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				coefs[i - 4] = constants::fits::fit[i][1]
					+ constants::fits::fit[i - 4][2]*t9i
					+ constants::fits::fit[i - 4][3]*t9i13
					+ constants::fits::fit[i - 4][4]*t913
					+ constants::fits::fit[i - 4][5]*t9
					+ constants::fits::fit[i - 4][6]*t953
					+ constants::fits::fit[i - 4][7]*lt;
				r(i, i - 1) = std::exp(coefs[i - 4]); 
			}

			Float val1=11.6045e0*t9i;
			Float val2=1.5e0*lt9;
			Float val3=val1*t9i*1.e-9;
			Float val4=1.5e-9*t9i;

			// Z <- Z "+ 1"
			for (int i = 4; i < 14; ++i) {
				int k = get_temperature_range(T);
				r(i - 1, i) = constants::fits::choose[i - 4][k]/constants::fits::choose[i + 1 - 4][k]*
					std::exp(
						  constants::fits::fit[i - 4][8]
						+ coefs[i - 4];
						- constants::fits::q  [i - 4]*val1
						+ val2
					);
			}


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				O <-> Ne
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// TODO !!!!!


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C <-> O
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// TODO !!!!!


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			more accurate computation of the rates of:
				C -> He
			!!!!!!!!!!!!!!!!!!!!!!!! */

			// TODO !!!!!


			return r;
		}

		template<typename Float>
		Eigen::Tensor<Float, 3> get_net14_fusion_rates(const Float T) {
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