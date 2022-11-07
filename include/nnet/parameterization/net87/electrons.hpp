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
 * @brief electron data reading and functions for net87.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <cmath>

#include "nnet_util/CUDA/cuda.inl"
#if COMPILE_DEVICE
#include "nnet_util/CUDA/cuda-util.hpp"
#endif

#include "nnet_util/eigen.hpp"

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

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

namespace nnet::net87::electrons
{
namespace constants
{
/*! @brief table initialize sucess */
extern bool table_read_success;

/*! @brief table sizes */
static const int nTemp = N_TEMP, nRho = N_RHO, nC = N_C;

DEVICE_DEFINE(extern double, log_temp_ref[N_TEMP], ;)
DEVICE_DEFINE(extern double, log_rho_ref[N_RHO], ;)
DEVICE_DEFINE(extern double, electron_rate[N_TEMP][N_RHO][N_C], ;)

/*! @brief read electron rate constants table for net87 */
bool inline read_cpu_table()
{
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

    return true;
}

bool inline copy_table_to_gpu()
{
#if COMPILE_DEVICE
    // copy to device
    gpuErrchk(cudaMemcpyToSymbol(dev_log_temp_ref, log_temp_ref, nTemp * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_log_rho_ref, log_rho_ref, nRho * sizeof(double), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_electron_rate, electron_rate, nTemp * nRho * nC * sizeof(double), 0,
                                 cudaMemcpyHostToDevice));
#endif
    return true;
}
} // namespace constants

/*! @brief interpolate electronic parameters
 *
 * @param temp     temperature
 * @param rhoElec  electron density
 * @param rate     electronic parameters to be populated
 */
template<typename Float>
HOST_DEVICE_FUN void inline interpolate(Float temp, Float rhoElec, std::array<Float, constants::nC>& rate)
{
    // find temperature index
    int   i_temp_sup = 0;
    Float log_temp   = std::log10(temp);
    while (i_temp_sup < constants::nTemp && constants::DEVICE_ACCESS(log_temp_ref)[i_temp_sup] < log_temp)
        ++i_temp_sup;

    // find rho index
    int   i_rho_sup = 0;
    Float log_rho   = std::log10(rhoElec);
    while (i_rho_sup < constants::nRho && constants::DEVICE_ACCESS(log_rho_ref)[i_rho_sup] < log_rho)
        ++i_rho_sup;

    // other limit index
    int i_temp_inf = std::max(0, i_temp_sup - 1);
    int i_rho_inf  = std::max(0, i_rho_sup - 1);
    i_temp_sup     = std::min(constants::nTemp - 1, i_temp_sup);
    i_rho_sup      = std::min(constants::nRho - 1, i_rho_sup);

    // distance between limits
    Float x2x = constants::DEVICE_ACCESS(log_temp_ref)[i_temp_sup] - log_temp;
    Float xx1 = -constants::DEVICE_ACCESS(log_temp_ref)[i_temp_inf] + log_temp;
    Float y2y = constants::DEVICE_ACCESS(log_rho_ref)[i_rho_sup] - log_rho;
    Float yy1 = -constants::DEVICE_ACCESS(log_rho_ref)[i_rho_inf] + log_rho;
    Float x2x1 =
        constants::DEVICE_ACCESS(log_temp_ref)[i_temp_sup] - constants::DEVICE_ACCESS(log_temp_ref)[i_temp_inf];
    x2x1       = x2x1 == 0 ? 2 : x2x1;
    Float y2y1 = constants::DEVICE_ACCESS(log_rho_ref)[i_rho_sup] - constants::DEVICE_ACCESS(log_rho_ref)[i_rho_inf];
    y2y1       = y2y1 == 0 ? 2 : y2y1;

    // actual interpolation
    for (int i = 0; i < constants::nC; ++i)
        rate[i] = (constants::DEVICE_ACCESS(electron_rate)[i_temp_inf][i_rho_inf][i] * x2x * y2y +
                   constants::DEVICE_ACCESS(electron_rate)[i_temp_sup][i_rho_inf][i] * xx1 * y2y +
                   constants::DEVICE_ACCESS(electron_rate)[i_temp_inf][i_rho_sup][i] * x2x * yy1 +
                   constants::DEVICE_ACCESS(electron_rate)[i_temp_sup][i_rho_sup][i] * xx1 * yy1) /
                  (x2x1 * y2y1);
}
} // namespace nnet::net87::electrons