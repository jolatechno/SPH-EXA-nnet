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

#ifdef USE_CUDA
	#include <cuda_runtime.h>

	#include "../CUDA/cuda-util.hpp"
#endif
#include "../CUDA/cuda.inl"

namespace nnet::net87::electrons {
	namespace constants {
		// table size
		static const int nTemp = N_TEMP, nRho = N_RHO, nC = N_C;

		// table type
		typedef eigen::fixed_size_array<double, nRho> rhoVector;
		typedef eigen::fixed_size_array<double, nTemp> tempVector;
		typedef eigen::fixed_size_array<double, nTemp*nRho*nC> rateMatrix;
		// typedef eigen::fixed_size_matrix<std::array<double, nC>, nTemp, nRho> rateMatrix; // double[nRho][nTemp][nC]

		CUDA_DEFINE(static double, log_temp_ref[N_TEMP], ;)
        CUDA_DEFINE(static double, log_rho_ref[N_RHO], ;)
        CUDA_DEFINE(static double, electron_rate[N_TEMP][N_RHO][N_C], ;)

		// read electron rate constants table
		bool read_table() {
			// read table
			const std::string electron_rate_table = { 
				#include ELECTRON_TABLE_PATH
			};

			// read file
	   		std::stringstream rate_table;
	   		rate_table << electron_rate_table;

			// read table
			for (int i = 0; i < nTemp; ++i)
				rate_table >> log_temp_ref[i];
			for (int i = 0; i < nRho; ++i)
				rate_table >> log_rho_ref[i];
			for (int i = 0; i < nTemp; ++i)
				for (int j = 0; j < nRho; ++j)
					for (int k = 0; k < nC; ++k)
						rate_table >> electron_rate[i][j][k];

#ifdef USE_CUDA
	        // copy to device 
			gpuErrchk(cudaMemcpyToSymbol(dev_log_temp_ref,  log_temp_ref,  nTemp*sizeof(double)));
	        gpuErrchk(cudaMemcpyToSymbol(dev_log_rho_ref,   log_rho_ref,   nRho*sizeof(double)));
	        gpuErrchk(cudaMemcpyToSymbol(dev_electron_rate, electron_rate, nTemp*nRho*nC*sizeof(double)));
#endif

			return true;
		}

		bool initalized = false;
#ifdef AUTO_INITIALIZE
		initalized = read_table();
#endif
	}


	struct interpolate_function {
		interpolate_function() {}

		/// interpolate electron rate
		/**
		 * TODO
		 */
		template<typename Float>
		CUDA_FUNCTION_DECORATOR void operator()(Float temp, Float rhoElec, std::array<double, constants::nC> &rate) const {
			// find temperature index
			int i_temp_sup = 0;
			Float log_temp = std::log10(temp);
			while (i_temp_sup < constants::nTemp && constants::CUDA_ACCESS(log_temp_ref)[i_temp_sup] < log_temp)
				++i_temp_sup;

			// find rho index
			int i_rho_sup = 0;
			Float log_rho = std::log10(rhoElec);
			while (i_rho_sup < constants::nRho && constants::CUDA_ACCESS(log_rho_ref)[i_rho_sup] < log_rho)
				++i_rho_sup;

			// other limit index
			int i_temp_inf = std::max(0,                    i_temp_sup - 1);
			int i_rho_inf  = std::max(0,                    i_rho_sup  - 1);
			    i_temp_sup = std::min(constants::nTemp - 1, i_temp_sup);
			    i_rho_sup  = std::min(constants::nRho  - 1, i_rho_sup);

			// distance between limits
			Float x2x  =  constants::CUDA_ACCESS(log_temp_ref)[i_temp_sup] - log_temp;
			Float xx1  = -constants::CUDA_ACCESS(log_temp_ref)[i_temp_inf] + log_temp;
			Float y2y  =  constants::CUDA_ACCESS(log_rho_ref )[i_rho_sup ] - log_rho;
			Float yy1  = -constants::CUDA_ACCESS(log_rho_ref )[i_rho_inf ] + log_rho;
			Float x2x1 =  constants::CUDA_ACCESS(log_temp_ref)[i_temp_sup] - constants::CUDA_ACCESS(log_temp_ref)[i_temp_inf];
				  x2x1 = x2x1 == 0 ? 2 : x2x1;
			Float y2y1 =  constants::CUDA_ACCESS(log_rho_ref )[i_rho_sup ] - constants::CUDA_ACCESS(log_rho_ref )[i_rho_inf ];
				  y2y1 = y2y1 == 0 ? 2 : y2y1;

			// actual interpolation
			for (int i = 0; i < constants::nC; ++i)
				rate[i] = (constants::CUDA_ACCESS(electron_rate)[i_temp_inf][i_rho_inf][i]*x2x*y2y
						+  constants::CUDA_ACCESS(electron_rate)[i_temp_sup][i_rho_inf][i]*xx1*y2y
						+  constants::CUDA_ACCESS(electron_rate)[i_temp_inf][i_rho_sup][i]*x2x*yy1
						+  constants::CUDA_ACCESS(electron_rate)[i_temp_sup][i_rho_sup][i]*xx1*yy1)/(x2x1*y2y1);
		}
	} interpolate;
}