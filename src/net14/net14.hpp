#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "../nuclear-net.hpp"
#include "net14-constants.hpp"

namespace nnet {
	namespace net14 {
		/// constant mass-excendent values
		const Eigen::VectorXd BE = Eigen::Vector<double, 14>(std::vector<double>{0.0, 7.27440, 14.43580, 19.16680, 28.48280, 38.46680, 45.41480, 52.05380, 59.09380, 64.22080, 71.91280, 79.85180, 87.84680, 90.55480}.data())*constants::Mev_to_cJ;

		/// ideal gaz correction
		Eigen::VectorXd ideal_gaz_correction(const double T) {
			Eigen::VectorXd BE_correction(14);
			for (int i = 0; i < 14; ++i) BE_correction(i) = -constants::Na * constants::Kb * T;
			return BE_correction;
		}

		// constant list of ordered reaction
		const std::vector<nnet::reaction> reaction_list = {
			/* !!!!!!!!!!!!!!!!!!!!!!!!
			fusions reactions from fits */
			{{{0}, {3}},  {{4}}},  // Ne + He -> Mg
			{{{0}, {4}},  {{5}}},  // Mg + He -> Si
			{{{0}, {5}},  {{6}}},  // Si + He -> S
			{{{0}, {6}},  {{7}}},  // S  + He -> Ar
			{{{0}, {7}},  {{8}}},  // Ar + He -> Ca
			{{{0}, {8}},  {{9}}},  // Ca + He -> Ti
			{{{0}, {9}},  {{10}}}, // Ti + He -> Cr
			{{{0}, {10}}, {{11}}}, // Cr + He -> Fe
			{{{0}, {11}}, {{12}}}, // Fe + He -> Ni
			{{{0}, {12}}, {{13}}}, // Ni + He -> Zn



			/* fission reactions from fits
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{4}},  {{0}, {3}}},  // Ne + He <- Mg
			{{{5}},  {{0}, {4}}},  // Mg + He <- Si
			{{{6}},  {{0}, {5}}},  // Si + He <- S
			{{{7}},  {{0}, {6}}},  // S  + He <- Ar
			{{{8}},  {{0}, {7}}},  // Ar + He <- Ca
			{{{9}},  {{0}, {8}}},  // Ca + He <- Ti
			{{{10}}, {{0}, {9}}},  // Ti + He <- Cr
			{{{11}}, {{0}, {10}}}, // Cr + He <- Fe
			{{{12}}, {{0}, {11}}}, // Fe + He <- Ni
			{{{13}}, {{0}, {12}}}, // Ni + He <- Zn



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			   3He -> C fusion */
			{{{0, 3}}, {{1}}},

			/* 3He <- C fission
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{1}}, {{0, 3}}},



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			2C -> Ne + He fusion
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{1, 2}}, {{3}, {0}}},

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			C + O -> Mg + He fusion
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{1}, {2}}, {{4}, {0}}},



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			2O -> Si + He fusion
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{2, 2}}, {{5}, {0}}},



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			   C + He -> O fusion */
			{{{0}, {1}}, {{2}}},

			/* C + He <- O fission
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{2}},  {{0}, {1}}},



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			   O + He -> Ne fusion */
			{{{0}, {2}}, {{3}}},

			/* O + He <- Ne fission
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{{{3}},  {{0}, {2}}}
		};

		/// compute a list of reactions for net14
		template<typename Float>
		std::vector<Float> compute_reaction_rates(const Float T) {
			std::vector<Float> rates;
			rates.reserve(reaction_list.size());

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			fusions and fissions reactions from fits
			!!!!!!!!!!!!!!!!!!!!!!!! */

			Float coefs[14 - 4];

			/* !!!!!!!!!!!!!!!!!!!!!!!!
			fusions reactions from fits */
			{
				const Float t9=T/1.e9;
				const Float t913=std::pow(t9, 1./3.);
				const Float t923=t913*t913;
				const Float t953=std::pow(t9, 5./3.);
				const Float t9i=1.e0/t9;
				const Float t9i2=t9i*t9i;
				const Float t9i13=1.e0/t913;
				const Float t9i23=t9i13*t9i13;
				const Float t9i43=t9i23*t9i23;
				const Float lt9=std::log(t9);

				/* fusion rates computed:
					- Ne + He -> Mg
					- Mg + He -> Si
					- Si + He -> S
					-  S + He -> Ar
					- Ar + He -> Ca
					- Ca + He -> Ti
					- Ti + He -> Cr
					- Cr + He -> Fe
					- Fe + He -> Ni
					- Ni + He -> Zn */
				for (int i = 4; i < 14; ++i) {
					coefs[i - 4] = 
						  constants::fits::fit[i - 4][1]*t9i
						+ constants::fits::fit[i - 4][2]*t9i13
						+ constants::fits::fit[i - 4][3]*t913
						+ constants::fits::fit[i - 4][4]*t9
						+ constants::fits::fit[i - 4][5]*t953
						+ constants::fits::fit[i - 4][6]*lt9;

					Float rate = std::exp(constants::fits::fit[i - 4][0] + coefs[i - 4]);
					rates.push_back(rate); 
				}
			}


			/* fission reactions from fits
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{
				const Float t9=T/1.e9;
				const Float t9i=1.e0/t9;
				const Float lt9=std::log(t9);

				const Float val1=11.6045e0*t9i;
				const Float val2=1.5e0*lt9;

				/* fision rates computed:
					- Ne + He <- Mg
					- Mg + He <- Si
					- Si + He <- S
					-  S + He <- Ar
					- Ar + He <- Ca
					- Ca + He <- Ti
					- Ti + He <- Cr
					- Cr + He <- Fe
					- Fe + He <- Ni
					- Ni + He <- Zn */
				for (int i = 4; i < 14; ++i) {
					int k = constants::fits::get_temperature_range(T);
					Float rate = constants::fits::choose[i - 4][k]/constants::fits::choose[i + 1 - 4][k]*
						std::exp(
							  coefs               [i - 4]
							+ constants::fits::fit[i - 4][7]
							- constants::fits::q  [i - 4]*val1
							+ val2
						);
					rates.push_back(rate);
				}
			}


			/* other fusion and fission reactions */
			{
				const Float t9r=T*1.0e-09;
	      		const Float t9=std::min((Float)10., t9r);
	      		const Float t92=t9*t9;
	      		const Float t93=t92*t9;
	      		const Float t95=t92*t93;
	      		const Float t912=std::sqrt(t9);
	      		const Float t913=std::pow(t9, 1./3.);
	      		const Float t923=t913*t913;
	      		const Float t932=t9*t912;
	      		const Float t943=t9*t913;
	      		const Float t953=t9*t923;

	      		const Float t9i=1./t9;
	      		const Float t9i13=1./t913;
	      	 	const Float t9i23=1./t923;
	      		const Float t9i32=1./t932;
	      		const Float t9rm1=1./t9r;
	      		const Float t9r32=t9*std::sqrt(t9r);

	      		const Float t9a=t9/(1. + 0.0396*t9);
			    const Float t9a13=std::pow(t9a, 1./3.);
			    const Float t9a56=std::pow(t9a, 5./6.);


			    /* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
			    {
					/* !!!!!!!!!!!!!!!!!!!!!!!!
				    3He -> C fusion */
				    const Float r2abe = (7.40e+05*t9i32)*std::exp(-1.0663*t9i)
						+ 4.164e+09*t9i23*std::exp(-13.49*t9i13-t92/0.009604)*(1. + 0.031*t913 + 8.009*t923 + 1.732*t9 + 49.883*t943 + 27.426*t953);
		      		Float r3a, rbeac = (130.*t9i32)*std::exp(-3.3364*t9i)
		      			+ 2.510e+07*t9i23*std::exp(-23.57*t9i13 - t92/0.055225)*(1. + 0.018*t913 + 5.249*t923 + 0.650*t9 + 19.176*t943 + 6.034*t953);
					if(T > 8e7) {
			      		r3a=2.90e-16*(r2abe*rbeac)
			      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
				    } else
			      		r3a=2.90e-16*(r2abe*rbeac)*(0.01 + 0.2*(1. + 4.*std::exp(-std::pow(0.025*t9i, 3.263)))/(1. + 4.*std::exp(-std::pow(t9/0.025, 9.227))))
			      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
					rates.push_back(r3a);


					/* 3He <- C fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
			      	const Float rev = 2.e20*std::exp(-84.419412e0*t9i);
			      	rates.push_back(r3a*rev*t93);
			    }

			    
				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2C -> Ne + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
		      		const Float r24=4.27e+26*t9a56*t9i32*std::exp(-84.165/t9a13 - 2.12e-03*t93);
		      		rates.push_back(r24);
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				C + O -> Mg + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{

					Float r1216=0.;
					if (T > 5e9) {
			            const Float t9ap=t9/(1. + 0.055*t9);
			            const Float t9a2p=t9ap*t9ap;
			            const Float t9a13p=std::pow(t9ap, 1./3.);
			            const Float t9a23p=t9a13*t9a13;
			            const Float t9a56ap=std::pow(t9ap, 5./6.);
			            r1216=1.72e+31*t9a56ap*t9i32*std::exp(-106.594/t9a13p)/(std::exp(-0.18*t9a2p) + 1.06e-03*std::exp(2.562*t9a23p));
			        }
			        rates.push_back(r1216);

				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2O -> Si + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					const Float r32=7.10d+36*t9i23*std::exp(-135.93*t9i13 - 0.629*t923 - 0.445*t943 + 0.0103*t92);
					rates.push_back(r32);
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
				{
					/* !!!!!!!!!!!!!!!!!!!!!!!!
					   C + He -> O fusion */
					const Float rcag = 1.04e+08/(t92*std::pow(1. + 0.0489*t9i23, 2.))*std::exp(-32.120*t9i13-t92/12.222016)
		            	+ 1.76e+08/(t92*std::pow(1. + 0.2654*t9i23, 2.))*std::exp(-32.120*t9i13)
		           			+ 1.25e+03*t9i32*std::exp(-27.499*t9i)
		           			+ 1.43e-02*t95*std::exp(-15.541*t9i);
					rates.push_back(rcag);


					/* C + He <- O fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
					const Float roga = rcag*5.13e+10*t9r32*std::exp(-83.108047*t9rm1);
					rates.push_back(roga);
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
				{
					/* !!!!!!!!!!!!!!!!!!!!!!!!
					   O + He -> Ne fusion */
					const Float roag=(9.37e+09*t9i23*std::exp(-39.757*t9i13-t92/2.515396) + 62.1*t9i32*std::exp(-10.297*t9i) + 538.*t9i32*std::exp(-12.226*t9i) + 13.*t92*std::exp(-20.093*t9i));
        			rates.push_back(roag);

					/* O + He <- Ne fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
					const Float rnega=roag*5.65e+10*t9r32*std::exp(-54.93807*t9rm1);
					rates.push_back(rnega);
				}
			}

			return rates;
		}
	}
}