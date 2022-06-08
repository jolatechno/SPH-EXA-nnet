#pragma once

#include <vector>
#include <iostream>

#include "../nuclear-net.hpp"
#include "../net86/net86.hpp"
#include "electrons.hpp"

namespace nnet::net87 {
	namespace constants = nnet::net86::constants;

	/// if true ignore coulombian corrections
	bool skip_coulombian_correction = false;

	/// constant mass-excendent values
	const std::vector<double> BE = [](){
		std::vector<double> BE_ = nnet::net86::BE;

		// electron energy
		BE_.push_back(0.782*constants::Mev_to_cJ);

		return BE_;
	}();

	/// constant list of ordered reaction
	const std::vector<nnet::reaction> reaction_list = []() {
		std::vector<nnet::reaction> reactions = nnet::net86::reaction_list;

		// electron captures
		reactions.push_back(nnet::reaction{{{constants::proton},  {constants::electron}}, {{constants::neutron}}});
		reactions.push_back(nnet::reaction{{{constants::neutron}, {constants::electron}}, {{constants::proton}}}); // assume position = electron

		return reactions;
	}();

	/// compute a list of rates for net87
	const auto compute_reaction_rates = [](const auto *Y, const auto T, const auto rho, const auto &eos_struct, auto *corrected_BE, auto *rates, auto *drates) {
		using Float = typename std::remove_const<decltype(T)>::type;

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		electron value
		!!!!!!!!!!!!!!!!!!!!!!!! */
		const Float Yelec   = Y[constants::electron];
		const Float rhoElec = Yelec*rho;
		std::array<Float, electrons::constants::nC> electron_values;
		electrons::interpolate(T, rhoElec, electron_values);

		Float effe        = electron_values[0];
		Float deffe       = electron_values[1]*1e-9;
		Float deffedYe    = electron_values[2];//*rho;
		Float Eneutr      = electron_values[3]*4.93e17;

		Float dEneutr     = electron_values[4]*4.93e17*1.e-9;
		Float dEneutrdYe  = electron_values[5]*4.93e17;//*rho

		Float effp        = electron_values[6];
		Float deffp       = electron_values[7];
		Float deffpdYe    = electron_values[8];//*rho;
		Float Eaneutr     = electron_values[9];
		Float dEaneutr    = electron_values[10];
		Float dEaneutrdYe = electron_values[11];//*rho;

		Float dUedYe = eos_struct.dU_dYe;


		nnet::net86::compute_reaction_rates(Y, T, rho, eos_struct, corrected_BE, rates, drates);

		/*********************************************/
		/* start computing the binding energy vector */
		/*********************************************/

		// ideal gaz correction
		Float kbt = constants::Kb*T;
		Float nakbt = constants::Na*kbt;
		Float correction = -1.5*nakbt;

		// adding electrons to net86
		corrected_BE[86] = BE.back() + correction;

		// electron energy corrections
		corrected_BE[constants::proton]  += Eneutr;
		corrected_BE[constants::neutron] += Eaneutr;

		/******************************************************/
		/* start computing reaction rate and their derivative */ 
		/******************************************************/

		int idx = 157-1 + 157-4 -1, jdx = 157-1 + 157-4 -1;
		// electron capture rates
		rates [++idx] = deffedYe; // = effe/rhoElec
		drates[++jdx] = rhoElec == 0 ? 0 : deffe/rhoElec;

		rates [++idx] = deffpdYe; // = deffp/rhoElec
		drates[++jdx] = rhoElec == 0 ? 0 : deffp/rhoElec; // deffp/Y[86]/rho, !!! hack !!!
	};
}