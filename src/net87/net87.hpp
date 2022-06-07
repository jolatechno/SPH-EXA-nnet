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

	/// function to compute the corrected BE
	const auto compute_BE = [](const auto T, const auto rho, const auto &eos_struct, auto *corrected_BE) {
		using Float = typename std::remove_const<decltype(T)>::type;

		// ideal gaz correction
		Float kbt = constants::Kb*T;
		Float nakbt = constants::Na*kbt;
		Float correction = -1.5*nakbt;

		// adding electrons to net86
		nnet::net86::compute_BE(T, rho, eos_struct, corrected_BE);
		corrected_BE[86] = BE.back() + correction;

		// electron energy corrections
		// BE_[constants::proton]  += Eneutr;
		// BE_[constants::neutron] += Eaneutr;

		//return corrected_BE;
	};

	/// constant list of ordered reaction
	const std::vector<nnet::reaction> reaction_list = []() {
		std::vector<nnet::reaction> reactions = nnet::net86::reaction_list;

		// electron captures
		reactions.push_back(nnet::reaction{{{constants::proton},  {constants::electron}}, {{constants::neutron}}});
		reactions.push_back(nnet::reaction{{{constants::neutron}, {constants::electron}}, {{constants::proton}}}); // assume position = electron

		return reactions;
	}();

	/// compute a list of rates for net87
	const auto compute_reaction_rates = [](const auto *Y, const auto T, const auto rho, const auto &eos_struct, auto *rates, auto *drates) {
		using Float = typename std::remove_const<decltype(T)>::type;

		nnet::net86::compute_reaction_rates(Y, T, rho, eos_struct, rates, drates);

		/* !!!!!!!!!!!!!!!!!!!!!!!!
		electron value
		!!!!!!!!!!!!!!!!!!!!!!!! */
		std::array<Float, electrons::constants::nC> electron_values;
		electrons::interpolate(T, rho*Y[86], electron_values);

		Float effe        = electron_values[0];
		Float deffe       = electron_values[1]    *1e-9;
		Float deffedYe    = electron_values[2]*rho;
		Float Eneutr      = electron_values[3]    *4.93e17;

		Float dEneutr     = electron_values[4]    *4.93e17*1.e-9;
		Float dEneutrdYe  = electron_values[5]*rho*4.93e17;

		Float effp        = electron_values[6];
		Float deffp       = electron_values[7];
		Float deffpdYe    = electron_values[8]*rho;
		Float Eaneutr     = electron_values[9];
		Float dEaneutr    = electron_values[10];
		Float dEaneutrdYe = electron_values[11]*rho;

		Float dUedYe = eos_struct.dU_dYe;

		int idx = 157-1 + 157-4 -1, jdx = 157-1 + 157-4 -1;
		// electron capture rates
		rates [++idx] = 0.; //deffedYe;            //  effe/Y[86]/rho, !!! hack !!!
		drates[++jdx] = 0.; //deffedYe*deffe/effe; // deffe/Y[86]/rho, !!! hack !!!

		rates [++idx] = 0.; //deffpdYe;            //  effp/Y[86]/rho, !!! hack !!!
		drates[++jdx] = 0.; //deffpdYe*deffp/effp; // deffp/Y[86]/rho, !!! hack !!!
	};
}