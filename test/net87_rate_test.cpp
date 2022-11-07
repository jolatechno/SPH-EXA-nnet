/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Simply prints out rates for net86/87.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include <iostream>

#include "nnet/parameterization/net87/net87.hpp"
#include "nnet/parameterization/eos/helmholtz.hpp"
#include "nnet/nuclear_net.hpp"

#define NET86_DEBUG

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::BE.size() << "\n\n";

#if DEBUG
	nnet::net86::debug = nnet::eos::debug = true;
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

	std::cout << "net14 <-> net87:\n";
	for (int i = 0; i < 14; ++i)
		std::cout << i << " <-> " << nnet::net87::constants::net14_species_order[i] << ", ";
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