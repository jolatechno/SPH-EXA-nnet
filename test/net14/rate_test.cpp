#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net14/net14.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net14::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net14::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net14::BE.size() << "\n\n";

#if NO_SCREENING
	nnet::net14::skip_coulombian_correction = true;
#endif

	// nnet::net14::skip_coulombian_correction = true;
	auto [rate, drates] = nnet::net14::compute_reaction_rates<double>(2e9, 1e9);

	std::cout << "reaction_list.size=" << nnet::net14::reaction_list.size() << ", rates.size=" << rate.size() << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	// print reactions
	std::cout << "\nrates (to sort...):\n";
	for (int i = 0; i < nnet::net14::reaction_list.size(); ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
}