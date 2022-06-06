#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net87/net87.hpp"
#include "../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::BE.size() << "\n\n";

#if NO_SCREENING
	nnet::net87::skip_coulombian_correction = true;
#endif
#if DEBUG
	nnet::debug = true;
#endif
	nnet::eos::helmholtz helm(nnet::net87::constants::Z);

	std::array<double, 86> Y, X;
    for (int i = 0; i < 86; ++i) X[i] = 0;
	X[nnet::net87::constants::net14_species_order[1]] = 0.5;
	X[nnet::net87::constants::net14_species_order[2]] = 0.5;
	for (int i = 0; i < 86; ++i) Y[i] = X[i]/nnet::net87::constants::A[i];

    std::vector<double> rate(nnet::net87::reaction_list.size()), drates(nnet::net87::reaction_list.size()), BE(87);
						  nnet::net87::compute_BE(               2e9, 1e9, BE.data());
	auto eos_struct     = helm(                               Y, 2e9, 1e9);
	                      nnet::net87::compute_reaction_rates(Y, 2e9, 1e9, eos_struct, rate, drates);

	std::cout << "reaction_list.size=" << nnet::net87::reaction_list.size() << ", rates.size=" << rate.size() << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	// print reactions
	std::cout << "\nspecial reaction rates:\n";
	for (int i = 0; i < 3; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
	std::cout << "(3He -> C)\t" << nnet::net87::reaction_list[3] << "\t\tR=" << rate[3] << ",\tdR/dT=" << drates[3] << "\n";
	std::cout << "(3He <- C)\t" << nnet::net87::reaction_list[4] << "\t\tR=" << rate[4] << ",\tdR/dT=" << drates[4] << "\n";

	std::cout << "\ndirect rates:\n";
	for (int i = 5; i < 157; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nreverse rates:\n";
	for (int i = 157; i < 157 + (157 - 5); ++i)
		std::cout << "(" << i-(157 - 5)+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nBE: ";
	for (int i = 0; i < 87; ++i)
		std::cout << BE[i] << ", ";
	std::cout << "\n";
}