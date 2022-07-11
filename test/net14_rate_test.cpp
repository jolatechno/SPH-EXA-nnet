#include <iostream>

#define NET14_DEBUG

#include "sph/traits.hpp"
#include "cstone/util/array.hpp"

#include "nnet/nuclear-net.cuh"
#include "nnet/net14/net14.cuh"
#include "nnet/eos/helmholtz.cuh"


int main() {
	std::cout << "A.size = " << nnet::net14::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net14::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net14::BE.size() << "\n\n";

	nnet::eos::helmholtz_constants::read_table<cstone::CpuTag>();

#if DEBUG
	nnet::net14::debug = nnet::eos::debug = true;
#endif

	std::array<double, 14> Y, X;
    for (int i = 0; i < 14; ++i) X[i] = 0;
	X[1] = 0.5;
	X[2] = 0.5;
    for (int i = 0; i < 14; ++i) Y[i] = X[i]/nnet::net14::constants::A[i];

   
    std::vector<double> rate(nnet::net14::reaction_list.size()), drates(nnet::net14::reaction_list.size()), BE(14);
	nnet::net14::compute_reaction_rates(Y.data(), 2e9, 1e9, NULL, BE.data(), rate.data(), drates.data());

	std::cout << "reaction_list.size=" << nnet::net14::reaction_list.size() << ", rates.size=" << rate.size() << "\n\n";
	
	int num_special_reactions = 5, num_reactions = 157 - 5, num_reverse = 157 - 5;

	std::cout << "\ndirect rates:\n";
	for (int i = 0; i < 16; ++i)
		std::cout << "(" << i+1 << ")\t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\ninverse rates:\n";
	for (int i = 16; i < 16 + 13; ++i)
		std::cout << "(" << i-15 << ")\t" << nnet::net14::reaction_list[i] << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";

	std::cout << "\nBE: ";
	for (int i = 0; i < 14; ++i)
		std::cout << BE[i] << ", ";
	std::cout << "\n";
}