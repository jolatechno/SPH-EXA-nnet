#include <iostream>

#include "../../src/nuclear-net.hpp"
#include "../../src/net87/net87.hpp"
#include "../../src/eos/helmholtz.hpp"

int main() {
	std::cout << "A.size = " << nnet::net87::constants::A.size() << "\n";
	std::cout << "Z.size = " << nnet::net87::constants::Z.size() << "\n";
	std::cout << "BE.size = " << nnet::net87::constants::BE.size() << "\n";
}