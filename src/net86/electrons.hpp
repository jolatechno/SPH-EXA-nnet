#pragma once

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

#include "../eigen/eigen.hpp"

#ifndef N_RHO
	#define N_RHO 541
#endif
#ifndef N_T
	#define N_TEMP 201
#endif
#ifndef N_C
	#define N_C 201
#endif
#ifndef ELECTRON_TABLE_PATH
	#define ELECTRON_TABLE_PATH "./electron_rate.dat"
#endif

#include <iostream>

#include <sstream>
#include <string>
#include <array>

namespace nnet::net86::electrons {
	namespace constants {
		// table size
		const int nRho = N_RHO, nTemp = N_TEMP, nC = N_C;

		// table type
		typedef eigen::fixed_size_matrix<std::array<double, nC>, nRho, nTemp> rateMatrix; // double[imax][jmax][nC]
		typedef std::array<double, nRho> rhoVector;
		typedef std::array<double, nTemp> tempVector;

		// read table
		const std::string rate_table = { 
			#include ELECTRON_TABLE_PATH
		};

		// read electron rate constants table
		std::tuple<rhoVector, tempVector, rateMatrix> read_table() {
			// table definitions
			rhoVector rho;
			tempVector temp;
			rateMatrix rates;

			// read table
			/* TODO */

			return {rho, temp, rates};
		}

		// tables
		auto const [rho_ref, temp_ref, electron_rate] = read_table();
	}
}