#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net87/net87.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::constants::BE.size() << "\n";

	std::cout << "\nComputing reaction rates...\n";
	auto [rate, drates] = nnet::net87::constants::compute_reaction_rates<double>(1e9, 1e9);
	std::cout << "\t...OK!\n\n";

	// print reactions
	for (int i = 0; i < nnet::net87::constants::reaction_list.size(); ++i) {
		auto const reaction = nnet::net87::constants::reaction_list[i];

		// print reactant
		for (auto [reactant_id, n_reactant_consumed] : reaction.reactants)
			std::cout << n_reactant_consumed << "*[" << reactant_id << "] ";

		std::cout << " ->  ";

		// print products
		for (auto [product_id, n_product_produced] : reaction.products)
			std::cout << n_product_produced << "*[" << product_id << "] ";

		std::cout << "\t\tR=" << rate[i] << ",\tdR/dT=" << drates[i] << "\n";
	}
}