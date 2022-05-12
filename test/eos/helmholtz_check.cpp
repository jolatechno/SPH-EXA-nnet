#include <iostream>

#include "../../src/eos/helmholtz.hpp"

int main() {
	std::cout << "d[2]=" << nnet::eos::helmholtz_constants::d[2] << "\n";
	std::cout << "t[4]=" << nnet::eos::helmholtz_constants::t[4] << "\n";
	std::cout << "f(2, 3)=" << nnet::eos::helmholtz_constants::f(2, 3) << "\n";
	std::cout << "fd(0, 0)=" << nnet::eos::helmholtz_constants::fd(0, 0) << "\n";

#if DEBUG
	nnet::eos::debug = true;
#endif

	std::vector<double> A = {12, 16}, Z = {6, 8}, Y = {.5/12, .5/16};
	nnet::eos::helmholtz eos(A, Z);

	auto res = eos(Y, 1e9, 1e9);
	std::cout << "\neos(...).cv=" << res.cv << "\n";
}