#pragma once

#include "../CUDA/cuda.inl"
#if COMPILE_DEVICE
	#include "../CUDA/cuda-util.hpp"
#endif

#include <vector>
#include <iostream>

#include "../nuclear-net.hpp"
#include "net86-constants.hpp"

#include "../sphexa/util/algorithm.hpp"

#define BE_NET86 \
	0 COMMA 0 COMMA \
	28.2970*constants::Mev_to_erg COMMA \
	92.1631*constants::Mev_to_erg COMMA \
	127.621*constants::Mev_to_erg COMMA \
	160.651*constants::Mev_to_erg COMMA \
	163.082*constants::Mev_to_erg COMMA \
	198.263*constants::Mev_to_erg COMMA \
	181.731*constants::Mev_to_erg COMMA \
	167.412*constants::Mev_to_erg COMMA \
	186.570*constants::Mev_to_erg COMMA \
	168.584*constants::Mev_to_erg COMMA \
	174.152*constants::Mev_to_erg COMMA \
	177.776*constants::Mev_to_erg COMMA \
	200.534*constants::Mev_to_erg COMMA \
	236.543*constants::Mev_to_erg COMMA \
	219.364*constants::Mev_to_erg COMMA \
	205.594*constants::Mev_to_erg COMMA \
	224.958*constants::Mev_to_erg COMMA \
	206.052*constants::Mev_to_erg COMMA \
	211.901*constants::Mev_to_erg COMMA \
	216.687*constants::Mev_to_erg COMMA \
	239.291*constants::Mev_to_erg COMMA \
	271.786*constants::Mev_to_erg COMMA \
	256.744*constants::Mev_to_erg COMMA \
	245.017*constants::Mev_to_erg COMMA \
	262.924*constants::Mev_to_erg COMMA \
	243.691*constants::Mev_to_erg COMMA \
	250.612*constants::Mev_to_erg COMMA \
	255.626*constants::Mev_to_erg COMMA \
	274.063*constants::Mev_to_erg COMMA \
	306.722*constants::Mev_to_erg COMMA \
	291.468*constants::Mev_to_erg COMMA \
	280.428*constants::Mev_to_erg COMMA \
	298.215*constants::Mev_to_erg COMMA \
	278.727*constants::Mev_to_erg COMMA \
	285.570*constants::Mev_to_erg COMMA \
	291.845*constants::Mev_to_erg COMMA \
	308.580*constants::Mev_to_erg COMMA \
	342.059*constants::Mev_to_erg COMMA \
	326.418*constants::Mev_to_erg COMMA \
	315.511*constants::Mev_to_erg COMMA \
	333.730*constants::Mev_to_erg COMMA \
	313.129*constants::Mev_to_erg COMMA \
	320.654*constants::Mev_to_erg COMMA \
	327.349*constants::Mev_to_erg COMMA \
	343.144*constants::Mev_to_erg COMMA \
	375.482*constants::Mev_to_erg COMMA \
	359.183*constants::Mev_to_erg COMMA \
	350.422*constants::Mev_to_erg COMMA \
	366.832*constants::Mev_to_erg COMMA \
	346.912*constants::Mev_to_erg COMMA \
	354.694*constants::Mev_to_erg COMMA \
	361.903*constants::Mev_to_erg COMMA \
	377.096*constants::Mev_to_erg COMMA \
	411.469*constants::Mev_to_erg COMMA \
	395.135*constants::Mev_to_erg COMMA \
	385.012*constants::Mev_to_erg COMMA \
	403.369*constants::Mev_to_erg COMMA \
	381.982*constants::Mev_to_erg COMMA \
	390.368*constants::Mev_to_erg COMMA \
	398.202*constants::Mev_to_erg COMMA \
	413.553*constants::Mev_to_erg COMMA \
	447.703*constants::Mev_to_erg COMMA \
	431.520*constants::Mev_to_erg COMMA \
	422.051*constants::Mev_to_erg COMMA \
	440.323*constants::Mev_to_erg COMMA \
	417.703*constants::Mev_to_erg COMMA \
	426.636*constants::Mev_to_erg COMMA \
	435.051*constants::Mev_to_erg COMMA \
	449.302*constants::Mev_to_erg COMMA \
	483.994*constants::Mev_to_erg COMMA \
	467.353*constants::Mev_to_erg COMMA \
	458.387*constants::Mev_to_erg COMMA \
	476.830*constants::Mev_to_erg COMMA \
	453.158*constants::Mev_to_erg COMMA \
	462.740*constants::Mev_to_erg COMMA \
	471.765*constants::Mev_to_erg COMMA \
	484.689*constants::Mev_to_erg COMMA \
	514.999*constants::Mev_to_erg COMMA \
	500.002*constants::Mev_to_erg COMMA \
	494.241*constants::Mev_to_erg COMMA \
	509.878*constants::Mev_to_erg COMMA \
	486.966*constants::Mev_to_erg COMMA \
	497.115*constants::Mev_to_erg COMMA \
	506.460*constants::Mev_to_erg

namespace nnet::net86 {
	extern bool debug;


#ifdef NET86_NO_COULOMBIAN_DEBUG
	/// if true ignore coulombian corrections
	const bool skip_coulombian_correction = true;
#else
	/// if true ignore coulombian corrections
	const bool skip_coulombian_correction = false;
#endif

	/// constant mass-excendent values
	DEVICE_DEFINE(inline static const std::array<double COMMA 86>, BE, = {
		BE_NET86
	};)
	

	/// constant list of ordered reaction
	inline static const nnet::reaction_list reaction_list = []() {
		nnet::reaction_list reactions;

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		2C fusion,
		C + O fusion,
		2O fusion
		!!!!!!!!!!!!!!!!!!!!!!!! */
		reactions.push_back(nnet::reaction{{{constants::main_reactant[0], 2}},                                  {{constants::main_product[0]}, {constants::alpha}}});
		reactions.push_back(nnet::reaction{{{constants::main_reactant[1]}, {constants::secondary_reactant[1]}}, {{constants::main_product[1]}, {constants::alpha}}});
		reactions.push_back(nnet::reaction{{{constants::main_reactant[2], 2}},                                  {{constants::main_product[2]}, {constants::alpha}}});

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		3He -> C fusion
		!!!!!!!!!!!!!!!!!!!!!!!! */
		reactions.push_back(nnet::reaction{{{constants::main_reactant[4], 3}}, {{constants::main_product[4]}}});

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		direct reaction
		!!!!!!!!!!!!!!!!!!!!!!!! */ 
		for (int i = 5; i < 157; ++i) {
			const int r1 = constants::main_reactant[i], r2 = constants::secondary_reactant[i], p = constants::main_product[i];

			int delta_Z = constants::Z[r1] + constants::Z[r2] - constants::Z[p], delta_A = constants::A[r1] + constants::A[r2] - constants::A[p];

			if (delta_Z == 0 && delta_A == 0) {
				reactions.push_back(nnet::reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}, {{constants::main_product[i]}}});
			} else if (delta_A == 1 && delta_Z == 0) {
				reactions.push_back(nnet::reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}, {{constants::main_product[i]}, {constants::neutron}}});
			} else if (delta_A == 1 && delta_Z == 1) {
				reactions.push_back(nnet::reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}, {{constants::main_product[i]}, {constants::proton}}});
			} else if (delta_A == 4 && delta_Z == 2) {
				reactions.push_back(nnet::reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}, {{constants::main_product[i]}, {constants::alpha}}});
			} else
				throw std::runtime_error("Mass conservation not possible when adding reaction to net86\n");
		}

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		C -> 3He fission
		!!!!!!!!!!!!!!!!!!!!!!!! */
		reactions.push_back(nnet::reaction{{{constants::main_product[4]}}, {{constants::main_reactant[4], 3}}});
		
		/* !!!!!!!!!!!!!!!!!!!!!!!!
		inverse reaction
		!!!!!!!!!!!!!!!!!!!!!!!! */
		for (int i = 5; i < 157; ++i) {
			const int r = constants::main_product[i], p1 = constants::main_reactant[i], p2 = constants::secondary_reactant[i];

			int delta_Z = constants::Z[r] - (constants::Z[p1] + constants::Z[p2]), delta_A = constants::A[r] - (constants::A[p1] + constants::A[p2]);

			if (delta_Z == 0 && delta_A == 0) {
				reactions.push_back(nnet::reaction{{{constants::main_product[i]}},                       {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
			} else if (delta_A == -1 && delta_Z ==  0) {
				reactions.push_back(nnet::reaction{{{constants::main_product[i]}, {constants::neutron}}, {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
			} else if (delta_A == -1 && delta_Z == -1) {
				reactions.push_back(nnet::reaction{{{constants::main_product[i]}, {constants::proton }}, {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
			} else if (delta_A == -4 && delta_Z == -2) {
				reactions.push_back(nnet::reaction{{{constants::main_product[i]}, {constants::alpha  }}, {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
			} else
				throw std::runtime_error("Mass conservation not possible when adding reaction to net86\n");	
		}

		return reactions;
	}();



	/// compute a list of rates for net86
	struct compute_reaction_rates_functor {
		compute_reaction_rates_functor() {}
		
		template<typename Float, class eos>
		HOST_DEVICE_FUN void inline operator()(const Float *Y, const Float T, const Float rho, const eos &eos_struct, Float *corrected_BE, Float *rates, Float *drates) const {
			/*********************************************/
			/* start computing the binding energy vector */
			/*********************************************/

			// ideal gaz correction
			const Float kbt = constants::Kb*T;
			const Float nakbt = constants::Na*kbt;
			const Float correction = -1.5*nakbt;

			for (int i = 0; i < 86; ++i)
				corrected_BE[i] = DEVICE_ACCESS(BE)[i] + correction;

			// coulombian correctio
			if (!skip_coulombian_correction) {
				const Float ne = rho*constants::Na/2.;
			    const Float ae = std::pow((3./4.)/(constants::pi*ne), 1./3.);
			    const Float gam = constants::e2/(kbt*ae);
			    for (int i = 0; i < 86; ++i) {
			    	const Float gamma = gam*std::pow(constants::DEVICE_ACCESS(Z)[i], 5./3.);
			    	const Float funcion = gamma > 1 ? constants::ggt1(gamma) : constants::glt1(gamma);

			    	//  if (debug) std::cout << "funcion[" << i << "]=" << funcion << (i == 13 ? "\n\n" : "\n");

				    corrected_BE[i] -= nakbt*funcion;
				}
			}


			/******************************************************/
			/* start computing reaction rate and their derivative */ 
			/******************************************************/


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			fusions and fissions reactions from fits
			!!!!!!!!!!!!!!!!!!!!!!!! */

			Float coefs[157 - 7], dcoefs[157 - 7],
				eff[157]={0.}, deff[157]={0.},
				l[157]={0.}, dl[157]={0.}, /* should be removed */
				mukbt[86]={0.}, deltamukbt[157]={0.};


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			fusions reactions rate and rate derivative from fits */
			{
				// constants:
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
				for (int i = 7; i < 157; ++i) {
					coefs[i - 7] = 
						  constants::fits::DEVICE_ACCESS(fit)[i - 7][1]*t9i
						+ constants::fits::DEVICE_ACCESS(fit)[i - 7][2]*t9i13
						+ constants::fits::DEVICE_ACCESS(fit)[i - 7][3]*t913
						+ constants::fits::DEVICE_ACCESS(fit)[i - 7][4]*t9
						+ constants::fits::DEVICE_ACCESS(fit)[i - 7][5]*t953
						+ constants::fits::DEVICE_ACCESS(fit)[i - 7][6]*lt9;

					dcoefs[i - 7] = (- constants::fits::DEVICE_ACCESS(fit)     [i - 7][1]*t9i2 
						             + (-   constants::fits::DEVICE_ACCESS(fit)[i - 7][2]*t9i43
						                +   constants::fits::DEVICE_ACCESS(fit)[i - 7][3]*t9i23
						                + 5*constants::fits::DEVICE_ACCESS(fit)[i - 7][5]*t923)*(1./3.)
						             + constants::fits::DEVICE_ACCESS(fit)     [i - 7][4]
						             + constants::fits::DEVICE_ACCESS(fit)     [i - 7][6]*t9i)*1e-9;

					eff[i] = std::exp(constants::fits::DEVICE_ACCESS(fit)[i - 7][0] + coefs[i - 7]);
					deff[i] = eff[i]*dcoefs[i - 7];


					// debuging :
#if !DEVICE_CODE
					if (debug)  { std::cout << "dir(" << i << ")=" << eff[i] << ", coef(" << i << ")=" << coefs[i - 7];
					              std::cout << "\tddir(" << i << ")=" << deff[i] << ", dcoef(" << i << ")=" << dcoefs[i - 7] << "\n"; }
#endif
				}
			}




			/* !!!!!!!!!!!!!!!!!!!!!!!!
			other fusion and fission reactions rate */
			{
				// constants:
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


			    /* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
			    {
					/* !!!!!!!!!!!!!!!!!!!!!!!!
				    3He -> C fusion */
				    const Float r2abe = (7.40e+05*t9i32)*std::exp(-1.0663*t9i)
						+ 4.164e+09*t9i23*std::exp(-13.49*t9i13-t92/0.009604)*(1. + 0.031*t913 + 8.009*t923 + 1.732*t9 + 49.883*t943 + 27.426*t953);
		      		Float rbeac = (130.*t9i32)*std::exp(-3.3364*t9i)
		      			+ 2.510e+07*t9i23*std::exp(-23.57*t9i13 - t92/0.055225)*(1. + 0.018*t913 + 5.249*t923 + 0.650*t9 + 19.176*t943 + 6.034*t953);
					if(T > 8e7) {
			      		eff[4]=2.90e-16*(r2abe*rbeac)
			      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);
				    } else
			      		eff[4] = 2.90e-16*(r2abe*rbeac)*(0.01 + 0.2*(1. + 4.*std::exp(-std::pow(0.025*t9i, 3.263)))/(1. + 4.*std::exp(-std::pow(t9/0.025, 9.227))))
			      			+ 0.1*1.35e-07*t9i32*std::exp(-24.811*t9i);


					/* 3He <- C fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
			      	const Float rev = 2.e20*std::exp(-84.419412e0*t9i);
			      	l[4] = eff[4]*rev*t93;


			      	// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "\nr3a=" << eff[4] << ", rg3a=" << l[4] << "\n";
#endif
			    }

			    
				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2C -> Ne + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					const Float t9a=t9/(1. + 0.0396*t9);
				    const Float t9a13=std::pow(t9a, 1./3.);
				    const Float t9a56=std::pow(t9a, 5./6.);

		      		eff[0] = 4.27e+26*t9a56*t9i32*std::exp(-84.165/t9a13 - 2.12e-03*t93);


		      		// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "r24=" << eff[0];
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				C + O -> Mg + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{	
					eff[1] = 0;
					if (T >= 5e8) {
			            const Float t9ap=t9/(1. + 0.055*t9);
			            const Float t9a2p=t9ap*t9ap;
			            const Float t9a13p=std::pow(t9ap, 1./3.);
			            const Float t9a23p=t9a13p*t9a13p;
			            const Float t9a56ap=std::pow(t9ap, 5./6.);
			            eff[1] = 1.72e+31*t9a56ap*t9i32*std::exp(-106.594/t9a13p)/(std::exp(-0.18*t9a2p) +
			            	1.06e-03*std::exp(2.562*t9a23p));
			        }


			        // debuging :
#if !DEVICE_CODE
					if (debug) std::cout << ", r1216=" << eff[1] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2O -> Si + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					eff[2]=7.10e+36*t9i23*std::exp(-135.93*t9i13 - 0.629*t923 - 0.445*t943 + 0.0103*t92);


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "r32=" << eff[2] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
				{
					/* !!!!!!!!!!!!!!!!!!!!!!!!
					   C + He -> O fusion */
					eff[5] = 1.04e+08/(t92*std::pow(1. + 0.0489*t9i23, 2.))*std::exp(-32.120*t9i13-t92/12.222016)
		            	+ 1.76e+08/(t92*std::pow(1. + 0.2654*t9i23, 2.))*std::exp(-32.120*t9i13)
		           			+ 1.25e+03*t9i32*std::exp(-27.499*t9i)
		           			+ 1.43e-02*t95*std::exp(-15.541*t9i);


					/* C + He <- O fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
					l[5] = eff[5]*5.13e+10*t9r32*std::exp(-83.108047*t9rm1);


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "rcag=" << eff[5] << ", roga=" << l[5] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				O + He <-> Ne fusion and fission
				!!!!!!!!!!!!!!!!!!!!!!!!*/
				{
					/* !!!!!!!!!!!!!!!!!!!!!!!!
					   O + He -> Ne fusion */
					eff[6]=(9.37e+09*t9i23*std::exp(-39.757*t9i13-t92/2.515396) + 62.1*t9i32*std::exp(-10.297*t9i) + 538.*t9i32*std::exp(-12.226*t9i) + 13.*t92*std::exp(-20.093*t9i));


					/* O + He <- Ne fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
					l[6] = eff[6]*5.65e+10*t9r32*std::exp(-54.93807*t9rm1);


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "roag=" << eff[6] << ", rnega=" << l[6] << "\n\n";
#endif
				}
			}


			/* other fusion and fission reactions rate derivative 
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{
				// constants:
				const Float t9=T*1.0e-09;
	  			const Float t92=t9*t9;
	  			const Float t93=t92*t9;
	  			const Float t94=t93*t9;
	  			const Float t95=t94*t9;
	  			const Float t912=std::sqrt(t9);
	  			const Float t913=std::pow(t9, 1./3.);
	  			const Float t923=t913*t913;
	  			const Float t932=t9*t912;
	  			const Float t943=t9*t913;
	  			const Float t952=t9*t932;
	  			const Float t953=t9*t923;
	  			const Float t9i=1./t9;
	  			const Float t9i2=t9i*t9i;
	  			const Float t9i3=t9i2*t9i;
	  			const Float t9i12=1./t912;
	  			const Float t9i13=1./t913;
	  			const Float t9i23=1./t923;
	  			const Float t9i32=1./t932;
	  			const Float t9i43=1./t943;
	  			const Float t9i52=1./t952;
	  			const Float t9i53=1./t953;


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				3He -> C fusion */
				{
					const Float vA=-24.811*t9i;
				    const Float vB=-1.0663*t9i;
				    const Float vC=-13.49*t9i13-t92*104.123282;
				    const Float vD=1. + .031*t913 + 8.009*t923 + 1.732*t9 + 49.883*t943 + 27.426*t953;
				    const Float vE=-3.3364*t9i;
				    const Float vF=-23.57*t9i13 - 18.10774106*t92;
				    const Float vG=1. + .018*t913 + 5.249*t923 + .650*t9+19.176*t943 + 6.034*t953;
				    const Float dvA=24.811*t9i2;
				    const Float dvB=1.0663*t9i2;
				    const Float dvC=t9i43*13.49/3. - 208.246564*t9;
				    const Float dvD=(t9i23*.031+t9i13*16.018 + 5.196 + t913*199.532 + t923*137.13)/3.;
				    const Float dvE=3.3364*t9i2;
				    const Float dvF=t9i43*23.57/3. - 36.21548212*t9;
				    const Float dvG=(t9i23*.018 + t9i13*10.498 + 1.950 + t913*76.704 + t923*30.17)/3.;

				      
				    const Float r2abe=7.4e5*t9i32*std::exp(vB) + 4.164e9*t9i23*vD*std::exp(vC);
				    const Float rbeac=130.*t9i32*std::exp(vE) + 2.510e7*t9i23*vG*std::exp(vF);
				    const Float dr2abe=std::exp(vB)*(-1.11e6*t9i52 + 7.4e5*t9i32*dvB) + 4.164e9*std::exp(vC)*(-t9i53*vD*2./3. + t9i23*dvC*vD + t9i23*dvD);
				    const Float drbeac=std::exp(vE)*(-195.*t9i52 + 130.*t9i32*dvE) + 2.510e7*std::exp(vF)*(-2.*t9i53*vG/3. + t9i23*dvF*vG + t9i23*dvG);
				    deff[4] =(2.90e-16*(dr2abe*rbeac + r2abe*drbeac) + 1.35e-8*std::exp(vA)*(-1.5*t9i52 + t9i32*dvA))*1.e-9;

			      	// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "\ndr3a=" << deff[4] << "\n";
#endif
		      	}


				/* 3He <- C fission
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					const Float vA=-84.419412*t9i;
				    const Float vB=std::pow(1.+.0489*t9i23, 2);
				    const Float vC=-32.120*t9i13 - std::pow(t9/3.496, 2.);
				    const Float vD=std::pow(1.+.2654*t9i23, 2);
				    const Float vE=-32.120*t9i13;
				    const Float vF=-27.499*t9i;
				    const Float vG=-15.541*t9i;
				    const Float dvA=84.419412*t9i2;
				    const Float dvB=-(2.+.0978*t9i23)*(.0326*t9i53);
				    const Float dvC=32.120*t9i43/3.-2.*t9/std::pow(3.496, 2.);
				    const Float dvD=-(2.+.5308*t9i23)*(.5308*t9i53/3.);
				    const Float dvE=32.120*t9i43/3.;
				    const Float dvF=27.499*t9i2;
				    const Float dvG=15.541*t9i2;

		      		dl[4] = 2.00e20*std::exp(vA)*t93*(dvA*eff[4] + 3.*t9i*eff[4] + deff[4])*1.e-9;


			      	// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "drg3a=" << dl[4] << "\n";
#endif
				}

			    
				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2C -> Ne + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					const Float vA=t9/(1. + .0396*t9);
				    const Float vA56=std::pow(vA, 5./6.);
				    const Float vB=-84.165*std::pow(vA, -1./3.) - 2.12e-3*t93;

				    const Float dvA=vA*vA*t9i2;
				    const Float dvB=28.055*dvA*std::pow(vA, -4./3.) - 6.36e-3*t92;

				    deff[0] = 4.27e26*t9i32*std::exp(vB)*(std::pow(vA, -1./6.)*dvA*5./6. - 1.5*vA56*t9i + vA56*dvB)*1.e-9;


		      		// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "dr24=" << deff[0] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				C + O -> Mg + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					if(t9 > .5) {
				        const Float vA=t9/(1. + .055*t9);
				        const Float vA56=std::pow(vA, 5./6.);
				        const Float vB=-106.594*std::pow(vA, -1./3.);
				        const Float vC=-.18*vA*vA;
				        const Float vD=2.562*std::pow(vA, 2./3.);
				        const Float val=std::exp(vC) + 1.06e-3*std::exp(vD);

				        const Float dvA=vA*vA*t9i2;
				        const Float dvB=106.594*std::pow(vA, -4./3.)*dvA/3.;
				        const Float dvC=-.36*vA*dvA;
				        const Float dvD=1.708*dvA*std::pow(vA, -1./3.);
				        const Float dval=dvC*std::exp(vC)+1.06e-3*dvD*std::exp(vD);


				        deff[1] = 1.72e31*t9i32*(std::pow(vA, -1./6.)*dvA*5./6. - 1.5*vA56*t9i+vA56*(dvB - dval/val))*std::exp(vB)/val*1.e-9;
				    }


			        // debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "dr1216=" << deff[1] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				2O -> Si + He fusion
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
					const Float vA=-135.93e0*t9i13 - .629*t923 - .445*t943 + .0103*t92;
					const Float dvA=45.31*t9i43 - .629*t9i13*2./3. - .445*t913*4./3. + .0206*t9;
					deff[2]=7.10e36*std::exp(vA)*t9i23*(-t9i*2./3. + dvA)*1.e-9;


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "dr32=" << deff[2] << "\n";
#endif
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				C + He -> O fusion */
				{
					const Float vA=-84.419412*t9i;
				    const Float vB=std::pow(1.+.0489*t9i23, 2);
				    const Float vC=-32.120*t9i13 - std::pow(t9/3.496, 2.);
				    const Float vD=std::pow(1.+.2654*t9i23, 2);
				    const Float vE=-32.120*t9i13;
				    const Float vF=-27.499*t9i;
				    const Float vG=-15.541*t9i;
				    const Float dvA=84.419412*t9i2;
				    const Float dvB=-(2.+.0978*t9i23)*(.0326*t9i53);
				    const Float dvC=32.120*t9i43/3.-2.*t9/std::pow(3.496, 2.);
				    const Float dvD=-(2.+.5308*t9i23)*(.5308*t9i53/3.);
				    const Float dvE=32.120*t9i43/3.;
				    const Float dvF=27.499*t9i2;
				    const Float dvG=15.541*t9i2;

					
		      		deff[5]=(1.04e8*std::exp(vC)*t9i2*(-2.*t9i + dvC - dvB/vB)/vB
	       				+ 1.76e8*std::exp(vE)*t9i2*(-2.*t9i + dvE - dvD/vD)/vD
	       				+ 1.25e3*std::exp(vF)*(-1.5*t9i52 + dvF*t9i32)
	       				+ 1.43e-2*std::exp(vG)*(5.*t94 + dvG*t95))*1.e-9;

		      		// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "drcag=" << deff[5] << "\n";
#endif
		      	}

				
				{
					const Float vA=-83.108047*t9i;
					const Float vB=-39.757*t9i13 - std::pow(t9/1.586, 2.);
					const Float vC=-10.297*t9i;
					const Float vD=-12.226*t9i;
					const Float vE=-20.093*t9i;
					const Float dvA=83.108047*t9i2;
					const Float dvB=39.757*t9i43/3. - 2.*t9/std::pow(1.586, 2.);
					const Float dvC=10.297*t9i2;
					const Float dvD=12.226*t9i2;
					const Float dvE=20.093*t9i2;

					/* C + He <- O fission
					!!!!!!!!!!!!!!!!!!!!!!!! */
					dl[5]=5.13e10*std::exp(vA)*(deff[5]*t932 + eff[5]*1.5*t912 + eff[5]*t932*dvA)*1.e-9;


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "droga=" << dl[5] << "\n";
#endif


					/* !!!!!!!!!!!!!!!!!!!!!!!!
					   O + He -> Ne fusion */
	    			deff[6] = (9.37e9*std::exp(vB)*(-t9i53*2./3. + t9i23*dvB)
			      		+ 62.1*std::exp(vC)*(-1.5*t9i52 + t9i32*dvC)
			      		+ 538.*std::exp(vD)*(-1.5*t9i52 + t9i32*dvD)
	       				+ 13.*std::exp(vE)*(2.*t9 + t92*dvE))*1.e-9;


		      		// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "droag=" << deff[6] << "\n";
#endif
				}


				/* O + He <- Ne fission
				!!!!!!!!!!!!!!!!!!!!!!!! */
				{
		      		const Float vA=-54.903255*t9i;
	  				const Float dvA=54.903255*t9i2;

	  				dl[6] = 5.65e10*std::exp(vA)*(deff[6]*t932 + 1.5*eff[6]*t912 + eff[6]*t932*dvA)*1.e-9;


					// debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "drnega=" << dl[6] << "\n\n";
#endif
				}
			}



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			compute reversed rates
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{
				const Float t9=T*1e-09;
				const Float t9i=1./t9;

				const Float val1 = 11.6045*t9i;
				const Float val2 = 1.5*std::log(t9);
				const Float val3 = val1*t9i*1e-9;
				const Float val4 = 1.5e-9*t9i;

				const int k = constants::fits::get_temperature_range(T);

				for (int i = 7; i < 137; ++i) {
					const Float part = constants::fits::DEVICE_ACCESS(choose)[constants::DEVICE_ACCESS(main_reactant)[i] - 5][k]/constants::fits::DEVICE_ACCESS(choose)[constants::DEVICE_ACCESS(main_product)[i] - 5][k];
					l[i]  = part*std::exp(constants::fits::DEVICE_ACCESS(fit)[i - 7][7] + coefs[i - 7] - val1*constants::fits::DEVICE_ACCESS(q)[i - 7] + val2);
			        dl[i] = l[i]*(dcoefs[i - 7] + val3*constants::fits::DEVICE_ACCESS(q)[i - 7] + val4);
				}
				// These are not photodesintegrations so they don't have val2
				for (int i = 137; i < 157; ++i) {
					const Float part = constants::fits::DEVICE_ACCESS(choose)[constants::DEVICE_ACCESS(main_reactant)[i] - 5][k]/constants::fits::DEVICE_ACCESS(choose)[constants::DEVICE_ACCESS(main_product)[i] - 5][k];
					l[i]  = part*std::exp(constants::fits::DEVICE_ACCESS(fit)[i - 7][7] + coefs[i - 7] - val1*constants::fits::DEVICE_ACCESS(q)[i - 7]);
			        dl[i] = l[i]*(dcoefs[i - 7] + val3*constants::fits::DEVICE_ACCESS(q)[i - 7]);
				}
			}



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			From most of the (a,p) and (a,n) we only have the inverse reaction.
			For that reason we use the inverse of the inverse as direct reaction.
			So the direct reactions from the tables are the inverse of our network.
			!!!!!!!!!!!!!!!!!!!!!!!! */
			for (int i = 137; i < 154; ++i) {
				algorithm::swap( eff[i],  l[i]);
				algorithm::swap(deff[i], dl[i]);
			}
			algorithm::swap( eff[156],  l[156]);
			algorithm::swap(deff[156], dl[156]);



			/* !!!!!!!!!!!!!!!!!!!!!!!!
			correction for direct rate for coulumbian correction
			!!!!!!!!!!!!!!!!!!!!!!!! */
			if (!skip_coulombian_correction) {
				/* !!!!!!!!!!!!!!!!!!!!!!!!
				compute deltamukbt */
				{
					const double ne  = rho*constants::Na/2.;
			    	const double kbt = T*constants::Kb;
			    	const double ae  = std::pow(3./(4.*constants::pi*ne), 1./3.);
			    	const double gam = constants::e2/(kbt*ae);

			    	const double a1 = -.898004;
			    	const double b1 = .96786;
			    	const double c1 = .220703;
			    	const double d1 = -.86097;
			    	const double e1 = 2.520058332;

			    	// compute mukbt
					for (int i = 0; i < 86; ++i) {
						const double gamp = gam*std::pow(constants::DEVICE_ACCESS(Z)[i], 5./3.);
				        const double sqrootgamp = std::sqrt(gamp);
				        const double sqroot2gamp = std::sqrt(sqrootgamp);
				        if(gamp <= 1) {
				            mukbt[i] = -(1./std::sqrt(3.))*gamp*sqrootgamp + std::pow(gamp, 1.9885)*.29561/1.9885;
				        } else
				            mukbt[i] = a1*gamp + 4.*(b1*sqroot2gamp - c1/sqroot2gamp) + d1*std::log(gamp) - e1;
					}

					// mu for neutrons must be zero
					mukbt[1] = 0;

					// compute deltamukbt
					for (int i = 0; i < 157; ++i)
						deltamukbt[i] = mukbt[constants::DEVICE_ACCESS(main_reactant)[i]] + mukbt[constants::DEVICE_ACCESS(secondary_reactant)[i]] - mukbt[constants::DEVICE_ACCESS(main_product)[i]];// - mukbt[constants::DEVICE_ACCESS(secondary_product)[i]];

					// Triple alpha correction
					deltamukbt[4] += mukbt[constants::DEVICE_ACCESS(main_reactant)[4]];

					// compute deltamukbt
					/*for (int i = 0; i < 157; ++i) {
						const auto &Reaction = reaction_list[i];

						Float deltamukbt_ = 0;
						for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants)
							deltamukbt_ += mukbt[reactant_id]*n_reactant_consumed;
						for (auto const [product_id, n_product_produced] : Reaction.products)
							deltamukbt_ -= mukbt[product_id]*n_product_produced;
						deltamukbt[i] = deltamukbt_ + mukbt[constants::alpha];
					}*/
				}


				/* !!!!!!!!!!!!!!!!!!!!!!!!
				correction for direct rate for coulumbian correction */
				for (int i = 0; i < 157; ++i) {
					Float EF = std::exp(deltamukbt[i]);
			        eff [i] =  eff[i]*EF;
			        deff[i] = deff[i]*EF - 2.*eff[i]*deltamukbt[i]/T;


			        // debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "EF[" << i << "]=" << EF << ", deltamukbt[" << i << "]=" << deltamukbt[i] << ", mukbt[" << i << "]=" << mukbt[i] << (i == 156 ? "\n\n" : "\n");
#endif
				}

				/* correction for inverse rate for coulumbian correction
				!!!!!!!!!!!!!!!!!!!!!!!! */
				for (int i = 137; i < 157; ++i) {
					Float EF = std::exp(deltamukbt[i]);
			        l [i] =  l[i]*EF;
			        dl[i] = dl[i]*EF - 2.*l[i]*deltamukbt[i]/T;


			        // debuging :
#if !DEVICE_CODE
					if (debug) std::cout << "EF[" << i << "]=" << EF << ", deltamukbt[" << i << "]=" << deltamukbt[i] << ", mukbt[" << i << "]=" << mukbt[i] << (i == 156 ? "\n\n" : "\n");
#endif
				}
			}
			


			/* !!!!!!!!!!!!!!!!!!!!!!!!
			push back rates
			!!!!!!!!!!!!!!!!!!!!!!!! */
			{
				int idx = -1, jdx = -1;

				/* !!!!!!!!!!!!!!!!!!!!!!!!
				push direct reaction rates
				!!!!!!!!!!!!!!!!!!!!!!!! */
				for (int i = 0; i < 157; ++i) 
					if (i != 3) {
						rates [++idx] =  eff[i];
						drates[++jdx] = deff[i];
					}

				/* !!!!!!!!!!!!!!!!!!!!!!!!
				push inverse reaction rates
				!!!!!!!!!!!!!!!!!!!!!!!! */
				for (int i = 4; i < 157; ++i) {
					rates [++idx] =  l[i];
					drates[++jdx] = dl[i];
				}
			}
		}
	};

	extern compute_reaction_rates_functor compute_reaction_rates;
}