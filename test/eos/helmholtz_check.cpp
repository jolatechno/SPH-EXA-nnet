#include <iostream>

#include "../../src/eos/helmholtz.hpp"

int main() {
	nnet::eos::helmholtz_constants::read_table("../../src/eos/helm_table.dat");
	std::cout << "d[2]=" << nnet::eos::helmholtz_constants::d[2] << "\n";
	std::cout << "t[4]=" << nnet::eos::helmholtz_constants::t[4] << "\n";
	std::cout << "f[2][3]=" << nnet::eos::helmholtz_constants::f[2][3] << "\n";
	std::cout << "fd[0][0]=" << nnet::eos::helmholtz_constants::fd[0][0] << "\n";

	std::vector<double> A = {2, 2}, Z = {1, 2}, Y = {.1, .2};
	nnet::eos::helmholtz eos(A, Z);

	auto res = eos(Y, 1e9, 1e9);
	std::cout << "\neos(...).cv=" << res.cv << "\n";
}