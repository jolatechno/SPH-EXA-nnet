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
#if DEBUG
	nnet::debug = true;
#endif

	auto [rate, drates] = nnet::net14::compute_reaction_rates<double>(2e9, 1e9);
	auto [BE, dBE_dT]   = nnet::net14::compute_BE<double>(            2e9, 1e9);

	std::cout << "reaction_list.size=" << nnet::net14::reaction_list.size() << ", rates.size=" << rate.size() << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	// print reactions
	std::cout << "\nspecial rates:\n";
	for (int i = 0; i < 5; ++i)
		std::cout << "   \t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
	
	std::cout << "\ndirect rates:\n";
	for (int i = 5; i < 5 + 12; ++i)
		std::cout << "(" << i-2 << ")\t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\ninverse rates:\n";
	for (int i = 5 + 12; i < 5 + 12 + 12; ++i)
		std::cout << "(" << i-2-12 << ")\t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nBE: ";
	for (int i = 0; i < 14; ++i)
		std::cout << BE[i] << ", ";
	std::cout << "\n";
}