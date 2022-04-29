#include <iostream>

#include "../../src/eos/helmholtz.hpp"

int main() {
	nnet::eos::helmotz_constants::read_table("../../src/eos/helm_table.dat");
	std::cout << "d[2]=" << nnet::eos::helmotz_constants::d[2] << "\n";
	std::cout << "t[4]=" << nnet::eos::helmotz_constants::t[4] << "\n";
	std::cout << "f[2][3]=" << nnet::eos::helmotz_constants::f[2][3] << "\n";
}