#pragma once

#include "../CUDA/cuda.inl"
#if COMPILE_DEVICE
	#include "../CUDA/cuda-util.hpp"
#endif

#include <vector>
#include <iostream>

#include "../nuclear-net.hpp"
#include "../net86/net86.hpp"
#include "electrons.hpp"

namespace nnet::net87 {
	namespace constants = nnet::net86::constants;

	/// constant mass-excendent values
	DEVICE_DEFINE(inline static const std::array<double COMMA 87>, BE, = {
		BE_NET86 COMMA
		0.782*constants::Mev_to_erg
	};)

	/// constant list of ordered reaction
	inline static const nnet::reaction_list reaction_list = []() {
		nnet::reaction_list reactions = nnet::net86::reaction_list;

		// electron captures
		reactions.push_back(nnet::reaction{{{constants::proton},  {constants::electron}}, {{constants::neutron}}});
		reactions.push_back(nnet::reaction{{{constants::neutron}, {constants::electron}}, {{constants::proton}}}); // assume position = electron

		return reactions;
	}();

	/// compute a list of rates for net87
	class compute_reaction_rates_functor {
	private:
		nnet::net86::compute_reaction_rates_functor net86_compute_reaction_rates;

	public:
		compute_reaction_rates_functor() {}

		template<typename Float, class eos>
		HOST_DEVICE_FUN void inline operator()(const Float *Y, const Float T, const Float rho, const eos &eos_struct, Float *corrected_BE, Float *rates, Float *drates) const {
			/* !!!!!!!!!!!!!!!!!!!!!!!!
			electron value
			!!!!!!!!!!!!!!!!!!!!!!!! */
			const Float Yelec   = Y[constants::electron];
			const Float rhoElec = Yelec*rho;
			std::array<Float, electrons::constants::nC> electron_values;
			electrons::interpolate(T, rhoElec, electron_values);

			Float effe        = electron_values[0];
			Float deffe       = electron_values[1];
			Float deffedYe    = electron_values[2];//*rho;
			Float Eneutr      = electron_values[3];

			Float dEneutr     = electron_values[4];
			Float dEneutrdYe  = electron_values[5];//rho;

			Float effp        = electron_values[6];
			Float deffp       = electron_values[7];
			Float deffpdYe    = electron_values[8];//*rho;
			Float Eaneutr     = electron_values[9];
			Float dEaneutr    = electron_values[10];
			Float dEaneutrdYe = electron_values[11];//*rho;

			Float dUedYe = eos_struct.dudYe;


			net86_compute_reaction_rates(Y, T, rho, eos_struct, corrected_BE, rates, drates);

			/*********************************************/
			/* start computing the binding energy vector */
			/*********************************************/

			// ideal gaz correction
			const Float kbt = constants::Kb*T;
			const Float nakbt = constants::Na*kbt;
			const Float correction = -1.5*nakbt;

			// adding electrons to net86
			corrected_BE[86] = DEVICE_ACCESS(BE).back() + correction;

			// electron energy corrections
			corrected_BE[constants::proton]  += -Eneutr;
			corrected_BE[constants::neutron] += -Eaneutr;

			/******************************************************/
			/* start computing reaction rate and their derivative */ 
			/******************************************************/

			int idx = 157-1 + 157-4 -1, jdx = 157-1 + 157-4 -1;
			// electron capture rates
			rates [++idx] = rhoElec == 0 ? 0 :  effe/rhoElec;
			drates[++jdx] = rhoElec == 0 ? 0 : deffe/rhoElec;

			rates [++idx] = rhoElec == 0 ? 0 :  effp/rhoElec;
			drates[++jdx] = rhoElec == 0 ? 0 : deffp/rhoElec;
		}
	};
	
	extern compute_reaction_rates_functor compute_reaction_rates;
}