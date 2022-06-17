#pragma once

#include <vector>
#include <iostream>

#include "../nuclear-net.hpp"
#include "../net86/net86.hpp"
#include "electrons.hpp"

#ifdef USE_CUDA
	#include <cuda_runtime.h>
#endif
#include "../CUDA/cuda.inl"

namespace nnet::net87 {
	namespace constants = nnet::net86::constants;

	/// if true ignore coulombian corrections
	bool skip_coulombian_correction = false;

	/// constant mass-excendent values
	CUDA_DEFINE(inline static const std::array<double COMMA 87>, BE, = {
		0 COMMA 0 COMMA
		28.296 *constants::Mev_to_cJ COMMA
		92.163 *constants::Mev_to_cJ COMMA
		127.621*constants::Mev_to_cJ COMMA
		160.651*constants::Mev_to_cJ COMMA
		163.082*constants::Mev_to_cJ COMMA
		198.263*constants::Mev_to_cJ COMMA
		181.731*constants::Mev_to_cJ COMMA
		167.412*constants::Mev_to_cJ COMMA
		186.570*constants::Mev_to_cJ COMMA
		168.584*constants::Mev_to_cJ COMMA
		174.152*constants::Mev_to_cJ COMMA
		177.776*constants::Mev_to_cJ COMMA
		200.534*constants::Mev_to_cJ COMMA
		236.543*constants::Mev_to_cJ COMMA
		219.364*constants::Mev_to_cJ COMMA
		205.594*constants::Mev_to_cJ COMMA
		224.958*constants::Mev_to_cJ COMMA
		206.052*constants::Mev_to_cJ COMMA
		211.901*constants::Mev_to_cJ COMMA
		216.687*constants::Mev_to_cJ COMMA
		239.291*constants::Mev_to_cJ COMMA
		271.786*constants::Mev_to_cJ COMMA
		256.744*constants::Mev_to_cJ COMMA
		245.017*constants::Mev_to_cJ COMMA
		262.924*constants::Mev_to_cJ COMMA
		243.691*constants::Mev_to_cJ COMMA
		250.612*constants::Mev_to_cJ COMMA
		255.626*constants::Mev_to_cJ COMMA
		274.063*constants::Mev_to_cJ COMMA
		306.722*constants::Mev_to_cJ COMMA
		291.468*constants::Mev_to_cJ COMMA
		280.428*constants::Mev_to_cJ COMMA
		298.215*constants::Mev_to_cJ COMMA
		278.727*constants::Mev_to_cJ COMMA
		285.570*constants::Mev_to_cJ COMMA
		291.845*constants::Mev_to_cJ COMMA
		308.580*constants::Mev_to_cJ COMMA
		342.059*constants::Mev_to_cJ COMMA
		326.418*constants::Mev_to_cJ COMMA
		315.511*constants::Mev_to_cJ COMMA
		333.730*constants::Mev_to_cJ COMMA
		313.129*constants::Mev_to_cJ COMMA
		320.654*constants::Mev_to_cJ COMMA
		327.349*constants::Mev_to_cJ COMMA
		343.144*constants::Mev_to_cJ COMMA
		375.482*constants::Mev_to_cJ COMMA
		359.183*constants::Mev_to_cJ COMMA
		350.422*constants::Mev_to_cJ COMMA
		366.832*constants::Mev_to_cJ COMMA
		346.912*constants::Mev_to_cJ COMMA
		354.694*constants::Mev_to_cJ COMMA
		361.903*constants::Mev_to_cJ COMMA
		377.096*constants::Mev_to_cJ COMMA
		411.469*constants::Mev_to_cJ COMMA
		395.135*constants::Mev_to_cJ COMMA
		385.012*constants::Mev_to_cJ COMMA
		403.369*constants::Mev_to_cJ COMMA
		381.982*constants::Mev_to_cJ COMMA
		390.368*constants::Mev_to_cJ COMMA
		398.202*constants::Mev_to_cJ COMMA
		413.553*constants::Mev_to_cJ COMMA
		447.703*constants::Mev_to_cJ COMMA
		431.520*constants::Mev_to_cJ COMMA
		422.051*constants::Mev_to_cJ COMMA
		440.323*constants::Mev_to_cJ COMMA
		417.703*constants::Mev_to_cJ COMMA
		426.636*constants::Mev_to_cJ COMMA
		435.051*constants::Mev_to_cJ COMMA
		449.302*constants::Mev_to_cJ COMMA
		483.994*constants::Mev_to_cJ COMMA
		467.353*constants::Mev_to_cJ COMMA
		458.387*constants::Mev_to_cJ COMMA
		476.830*constants::Mev_to_cJ COMMA
		453.158*constants::Mev_to_cJ COMMA
		462.740*constants::Mev_to_cJ COMMA
		471.765*constants::Mev_to_cJ COMMA
		484.689*constants::Mev_to_cJ COMMA
		514.999*constants::Mev_to_cJ COMMA
		500.002*constants::Mev_to_cJ COMMA
		494.241*constants::Mev_to_cJ COMMA
		509.878*constants::Mev_to_cJ COMMA
		486.966*constants::Mev_to_cJ COMMA
		497.115*constants::Mev_to_cJ COMMA
		506.460*constants::Mev_to_cJ COMMA
		0.782*constants::Mev_to_cJ
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
	class compute_reaction_rates_function {
	private:
		electrons::interpolate_function electron_interpolate;

	public:
		compute_reaction_rates_function() {}

		template<typename Float, class eos>
		CUDA_FUNCTION_DECORATOR void inline operator()(const Float *Y, const Float T, const Float rho, const eos &eos_struct, Float *corrected_BE, Float *rates, Float *drates) const {
			/* !!!!!!!!!!!!!!!!!!!!!!!!
			electron value
			!!!!!!!!!!!!!!!!!!!!!!!! */
			const Float Yelec   = Y[constants::electron];
			const Float rhoElec = Yelec*rho;
			std::array<Float, electrons::constants::nC> electron_values;
			electron_interpolate(T, rhoElec, electron_values);

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
			corrected_BE[86] = CUDA_ACCESS(BE).back() + correction;

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
		}
	} compute_reaction_rates;
}