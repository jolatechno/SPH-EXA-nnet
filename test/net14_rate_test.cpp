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
 * @brief Simply prints out rates for net14.
 */

#include <iostream>

#define NET14_DEBUG

#include "sph/traits.hpp"
#include "cstone/util/array.hpp"

#include "nnet/nuclear-net.hpp"
#include "nnet/net14/net14.hpp"
#include "nnet/eos/helmholtz.hpp"


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