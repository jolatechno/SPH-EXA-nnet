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
 * @brief ideal gas EOS definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "nnet_util/CUDA/cuda.inl"

namespace nnet::eos
{
namespace ideal_gas_constants
{
const static double Kb = 1.380658e-16;
const static double Na = 6.022137e23;
const static double R  = 8.317e7;
} // namespace ideal_gas_constants

/*! @brief Ideal gas functor class */
template<typename Float>
class ideal_gas_functor : public nnet::eos_functor<Float>
{
private:
    double mu;

public:
    ideal_gas_functor(double mu_)
        : mu(mu_)
    {
    }

    HOST_DEVICE_FUN ideal_gas_functor() {}
    HOST_DEVICE_FUN ~ideal_gas_functor() {}

    /*! @brief Ideal gas EOS for nuclear networks.
     *
     * @param Y    molar proportions
     * @param T    temperature
     * @param rho  density
     *
     * Returns ideal gas EOS output struct.
     */
    HOST_DEVICE_FUN nnet::eos_struct<Float> inline operator()(const Float* Y, const Float T,
                                                              const Float rho) const override
    {
        nnet::eos_struct<Float> res;

        const Float dmy = ideal_gas_constants::R / mu;
        res.cv          = 1.5 * dmy;
        res.u           = T * res.cv;
        res.p           = rho * T * dmy;
        res.c           = std::sqrt(res.p / rho);
        res.dpdT        = rho * dmy;

        return res;
    }
};
} // namespace nnet::eos