#include <iostream>

#include "../src/nuclear-net.hpp"
#include "../src/net87/net87.hpp"
#include "../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::BE.size() << "\n\n";

	if (!nnet::eos::helmholtz_constants::initalized)
		nnet::eos::helmholtz_constants::initalized    = nnet::eos::helmholtz_constants::read_table();
	if (!nnet::net87::electrons::constants::initalized)
		nnet::net87::electrons::constants::initalized = nnet::net87::electrons::constants::read_table();

#if NO_SCREENING
	nnet::net87::skip_coulombian_correction = true;
#endif
#if DEBUG
	nnet::debug = true;
#endif
	nnet::eos::helmholtz_functor helm(nnet::net87::constants::Z);

	std::array<double, 87> Y, X;
    for (int i = 0; i < 87; ++i) X[i] = 0;
	X[nnet::net87::constants::net14_species_order[1]] = 0.5;
	X[nnet::net87::constants::net14_species_order[2]] = 0.5;
	for (int i = 0; i < 86; ++i) Y[i] = X[i]/nnet::net87::constants::A[i];
	Y[86] = 1;

    std::vector<double> rate(nnet::net87::reaction_list.size()), drates(nnet::net87::reaction_list.size()), BE(87);
	auto eos_struct = helm(                               Y.data(), 3e9, 1e9);
	                  nnet::net87::compute_reaction_rates(Y.data(), 3e9, 1e9, eos_struct, BE.data(), rate.data(), drates.data());

	"net14 <-> net87:\t";
	for (int i = 0; i < 14; ++i)
		std::cout << i << " <-> " << nnet::net87::constants::net14_species_order[i] << ",\t";
	std::cout << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	// print reactions
	std::cout << "\ndirect rates:\n";
	for (int i = 0; i < 157-1; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nreverse rates:\n";
	for (int i = 157-1; i < 157-1 + 157-4; ++i)
		std::cout << "(" << i-(157 - 5)+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nelectron capture rates:\n";
	for (size_t i = 157-1 + 157-4; i < nnet::net87::reaction_list.size(); ++i)
		std::cout << "(" << i-(157 - 5)+1 << ")\t" << nnet::net87::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nBE: ";
	for (int i = 0; i < 87; ++i)
		std::cout << BE[i] << ", ";
	std::cout << "\n";
}