#pragma once

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

#include "../eigen/eigen.hpp"

#ifndef N_TEMP
	#define N_TEMP 41
#endif
#ifndef N_RHO
	#define N_RHO 51
#endif
#ifndef N_C
	#define N_C 12
#endif
#ifndef ELECTRON_TABLE_PATH
	#define ELECTRON_TABLE_PATH "./electron_rate.dat"
#endif

#include <iostream>

#include <sstream>
#include <string>
#include <array>

#include <cmath>

namespace nnet::net86::electrons {
	namespace constants {
		// table size
		const int nTemp = N_TEMP, nRho = N_RHO, nC = N_C;

		// table type
		typedef eigen::fixed_size_matrix<std::array<double, nC>, nTemp, nRho> rateMatrix; // double[nRho][nTemp][nC]
		typedef std::array<double, nRho> rhoVector;
		typedef std::array<double, nTemp> tempVector;

		// read table
		const std::string electron_rate_table = { 
			#include ELECTRON_TABLE_PATH
		};

		// read electron rate constants table
		std::tuple<tempVector, rhoVector, rateMatrix> read_table() {
			// read file
	   		std::stringstream rate_table;
	   		rate_table << electron_rate_table;

			// table definitions
			tempVector temp;
			rhoVector rho;
			rateMatrix rates;

			// read table
			for (int i = 0; i < nTemp; ++i)
				rate_table >> temp[i];
			for (int i = 0; i < nRho; ++i)
				rate_table >> rho[i];
			for (int i = 0; i < nTemp; ++i)
				for (int j = 0; j < nRho; ++j)
					for (int k = 0; k < nC; ++k)
						rate_table >> rates(i,j)[k];

			return {temp, rho, rates};
		}

		// tables
		auto const [log_temp_ref, log_rho_ref, electron_rate] = read_table();
	}

	namespace util {
		/// interpolate electron rate
		/**
		 * TODO
		 */
		template<typename Float>
		void interpolate(Float temp, Float rhoElec, std::array<double, constants::nC> &rate) {
			// find temperature index
			int i_temp_sup = 1;
			Float log_temp = std::log10(temp);
			for (; i_temp_sup < constants::nTemp &&
				constants::log_temp_ref[i_temp_sup] > log_temp;
				++i_temp_sup) {}

			// find rho index
			int i_rho_sup = 1;
			Float log_rho = std::log10(rhoElec);
			for (; i_rho_sup < constants::nRho &&
				constants::log_rho_ref[i_rho_sup] > log_rho;
				++i_rho_sup) {}

			// other limit index
			int i_temp_inf = i_temp_sup - 1;
			int i_rho_inf  = i_rho_sup - 1;

			// distance between limits
			Float x2x  =  constants::log_temp_ref[i_temp_sup] - log_temp;
			Float xx1  = -constants::log_temp_ref[i_temp_inf] + log_temp;
			Float y2y  =  constants::log_rho_ref[i_rho_sup] - log_rho;
			Float yy1  = -constants::log_rho_ref[i_rho_inf] + log_rho;
			Float x2x1 = constants::log_temp_ref[i_temp_sup] - constants::log_temp_ref[i_temp_inf];
			Float y2y1 = constants::log_rho_ref[i_rho_sup]   - constants::log_rho_ref[i_rho_inf];

			// actual interpolation
			for (int i = 0; i < constants::nC; ++i)
				rate[i] = (constants::electron_rate(i_temp_inf, i_rho_inf)[i]*x2x*y2y
						+  constants::electron_rate(i_temp_sup, i_rho_inf)[i]*xx1*y2y
						+  constants::electron_rate(i_temp_inf, i_rho_sup)[i]*x2x*yy1
						+  constants::electron_rate(i_temp_sup, i_rho_sup)[i]*xx1*yy1)/(x2x1*y2y1);
		}
	}
}