#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net87/net87.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::constants::BE.size() << "\n\n";

	nnet::net87::constants::skip_coulombian_correction = true;
	auto [rate, drates] = nnet::net87::constants::compute_reaction_rates<double>(2e9, 1e9);

	std::cout << "reaction_list.size=" << nnet::net87::constants::reaction_list.size() << ", rates.size=" << rate.size() << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	// print reactions
	std::cout << "\nspecial reaction rates:\n";
	for (int i = 0; i < 3; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net87::constants::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
	std::cout << "(3He -> C)\t" << nnet::net87::constants::reaction_list[3] << "\t\tR=" << rate[3] << ",\tdR/dT=" << drates[3] << "\n";
	std::cout << "(3He <- C)\t" << nnet::net87::constants::reaction_list[4] << "\t\tR=" << rate[4] << ",\tdR/dT=" << drates[4] << "\n";

	std::cout << "\ndirect rates:\n";
	for (int i = 5; i < 157; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net87::constants::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nreverse rates:\n";
	for (int i = 157; i < 157 + (157 - 5); ++i)
		std::cout << "(" << i-(157 - 5)+1 << ")\t" << nnet::net87::constants::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
}