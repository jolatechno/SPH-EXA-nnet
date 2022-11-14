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
 * @brief Helmholtz EOS definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <tuple>
#include <math.h>

#include "nnet_util/CUDA/cuda.inl"
#if COMPILE_DEVICE
#include "nnet_util/CUDA/cuda-util.hpp"
#endif

#include "nnet/nuclear_net.hpp"

#include "nnet_util/eigen.hpp"

#include "nnet_util/algorithm.hpp"

#ifndef IMAX
#define IMAX 541
#endif
#ifndef JMAX
#define JMAX 201
#endif
#ifndef HELM_TABLE_PATH
#define HELM_TABLE_PATH "./helm_table.dat"
#endif

namespace nnet::eos
{

namespace helmholtz
{

/*! @brief if true print debuging prints */
extern bool debug;

namespace constants
{
/*! @brief table initialize sucess */
extern bool table_read_success;

// table size
const int imax = IMAX, jmax = JMAX;

// table limits
const double tlo   = 3.;
const double thi   = 13.;
const double tstp  = (thi - tlo) / (double)(jmax - 1);
const double tstpi = 1. / tstp;
const double dlo   = -12.;
const double dhi   = 15.;
const double dstp  = (dhi - dlo) / (double)(imax - 1);
const double dstpi = 1. / dstp;

// physical constants
const double pi =
    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420198938095257201065485863278865936153381827968230301952035301852968995773622599413891249721775283479131515574857242454150695950829533116861727855889075098381754637464939319;
const double g       = 6.6742867e-8;
const double h       = 6.6260689633e-27;
const double hbar    = 0.5 * h / pi;
const double qe      = 4.8032042712e-10;
const double avo     = 6.0221417930e23;
const double clight  = 2.99792458e10;
const double kerg    = 1.380650424e-16;
const double ev2erg  = 1.60217648740e-12;
const double kev     = kerg / ev2erg;
const double amu     = 1.66053878283e-24;
const double mn      = 1.67492721184e-24;
const double mp      = 1.67262163783e-24;
const double me      = 9.1093821545e-28;
const double rbohr   = hbar * hbar / (me * qe * qe);
const double fine    = qe * qe / (hbar * clight);
const double hion    = 13.605698140;
const double ssol    = 5.6704e-5;
const double asol    = 4.0 * ssol / clight;
const double weinlam = h * clight / (kerg * 4.965114232);
const double weinfre = 2.821439372 * kerg / h;
const double rhonuc  = 2.342e14;
const double kergavo = kerg * avo;
const double sioncon = (2.0 * pi * amu * kerg) / (h * h);

// parameters
const double a1   = -0.898004;
const double b1   = 0.96786;
const double c1   = 0.220703;
const double d1   = -0.86097;
const double e1   = 2.5269;
const double a2   = 0.29561;
const double b2   = 1.9885;
const double c2   = 0.288675;
const double esqu = qe * qe;

/*! @brief Read helmholtz constant table. */
extern bool readCPUTable();
extern bool copyTableToGPU();

} // namespace constants

/*! @brief Helmholtz EOS
 *
 * @param abar_ average A (number of mass)
 * @param zbar_ average Z (number of charge)
 * @param temp  temperature
 * @param rho   density
 *
 * Returns Helmholtz EOS output struct.
 */
template<typename Float>
extern HOST_DEVICE_FUN nnet::eos_struct<Float> helmholtzEos(double abar_, double zbar_, const Float temp,
                                                            const Float rho);

} // namespace helmholtz

/*! @brief Helmholtz functor class */
template<typename Float>
class HelmholtzFunctor : public nnet::EosFunctor<Float>
{
private:
    const Float* Z;
    int          dimension;
#if COMPILE_DEVICE
    Float* devZ;
#endif

public:
    HelmholtzFunctor(const Float* Z_, int dimension_)
        : Z(Z_)
        , dimension(dimension_)
    {
#if COMPILE_DEVICE
        gpuErrchk(cudaMalloc(&devZ, dimension * sizeof(Float)));
        gpuErrchk(cudaMemcpy(devZ, Z, dimension * sizeof(Float), cudaMemcpyHostToDevice));
#endif
    }
    template<class Vector>
    HelmholtzFunctor(const Vector& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    template<class Vector>
    HelmholtzFunctor(const Vector& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    HelmholtzFunctor(const std::vector<Float>& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    HelmholtzFunctor(const std::vector<Float>& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    template<size_t n>
    HelmholtzFunctor(const std::array<Float, n>& Z_, int dimension_)
        : HelmholtzFunctor(Z_.data(), dimension_)
    {
    }
    template<size_t n>
    HelmholtzFunctor(const std::array<Float, n>& Z_)
        : HelmholtzFunctor(Z_.data(), Z_.size())
    {
    }

    HOST_DEVICE_FUN HelmholtzFunctor() {}
    HOST_DEVICE_FUN ~HelmholtzFunctor()
    {
#if COMPILE_DEVICE && !DEVICE_CODE
        cudaFree(devZ);
#endif
    }

    /*! @brief Ideal gas EOS for nuclear networks.
     *
     * @param Y    molar proportions
     * @param temp temperature
     * @param rho  density
     *
     * Returns Helmholtz EOS output struct.
     */
    HOST_DEVICE_FUN nnet::eos_struct<Float> inline operator()(const Float* Y, const Float temp,
                                                              const Float rho) const override
    {
        // compute abar and zbar
        double abar = algorithm::accumulate(Y, Y + dimension, (double)0.);
        double zbar = eigen::dot(Y, Y + dimension, DEVICE_ACCESS(Z));

        return helmholtz::helmholtzEos(abar, zbar, temp, rho);
    }
};
} // namespace nnet::eos