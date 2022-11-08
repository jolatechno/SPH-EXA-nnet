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
 * @brief net87 definitions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <vector>
#include <iostream>

#include "nnet_util/CUDA/cuda.inl"
#if COMPILE_DEVICE
#include "nnet_util/CUDA/cuda-util.hpp"
#endif

#include "nnet/nuclear_net.hpp"

#include "../net86/net86.hpp"

#include "electrons.hpp"

namespace nnet::net87
{
namespace constants = nnet::net86::constants;

/*! @brief constant mass-excendent values */
DEVICE_DEFINE(inline static const std::array<double COMMA 87>, BE, = {BE_NET86 COMMA 0.782 * constants::MevToErg};)

/*! @brief constant list of ordered reaction */
inline static const nnet::ReactionList reactionList = []()
{
    nnet::ReactionList reactions = nnet::net86::reactionList;

    // electron captures
    reactions.pushBack(nnet::Reaction{{{constants::proton}, {constants::electron}}, {{constants::neutron}}});
    reactions.pushBack(nnet::Reaction{{{constants::neutron}, {constants::electron}},
                                      {{constants::proton}}}); // assume position = electron

    return reactions;
}();

/*! @brief net87 functor */
template<typename Float>
class ComputeReactionRatesFunctor : public nnet::ComputeReactionRatesFunctor<Float>
{
private:
    nnet::net86::ComputeReactionRatesFunctor<Float> net86ComputeReactionRates;

public:
    ComputeReactionRatesFunctor() {}

    /*! @brief compute net87 rates
     *
     * @param Y              molar fractions
     * @param T             temperature
     * @param rho           density
     * @param eos_struct    eos struct to populate
     * @param corrected_BE  will be populated by binding energies, corrected by coulombien terms
     * @param rates         will be populated with reaction rates
     * @param drates        will be populated with the temperature derivatives of reaction rates
     */
    HOST_DEVICE_FUN void inline operator()(const Float* Y, const Float T, const Float rho,
                                           const nnet::eos_struct<Float>& eos_struct, Float* corrected_BE, Float* rates,
                                           Float* drates) const override
    {
        /* !!!!!!!!!!!!!!!!!!!!!!!!
        electron value
        !!!!!!!!!!!!!!!!!!!!!!!! */
        const Float                                 Yelec   = Y[constants::electron];
        const Float                                 rhoElec = Yelec * rho;
        std::array<Float, electrons::constants::nC> electron_values;
        electrons::interpolate(T, rhoElec, electron_values);

        Float effe     = electron_values[0];
        Float deffe    = electron_values[1];
        Float deffedYe = electron_values[2]; //*rho;
        Float Eneutr   = electron_values[3];

        Float dEneutr    = electron_values[4];
        Float dEneutrdYe = electron_values[5]; // rho;

        Float effp        = electron_values[6];
        Float deffp       = electron_values[7];
        Float deffpdYe    = electron_values[8]; //*rho;
        Float Eaneutr     = electron_values[9];
        Float dEaneutr    = electron_values[10];
        Float dEaneutrdYe = electron_values[11]; //*rho;

        Float dUedYe = eos_struct.dudYe;

        net86ComputeReactionRates(Y, T, rho, eos_struct, corrected_BE, rates, drates);

        /*********************************************/
        /* start computing the binding energy vector */
        /*********************************************/

        // ideal gaz correction
        const Float kbt        = constants::Kb * T;
        const Float nakbt      = constants::Na * kbt;
        const Float correction = -1.5 * nakbt;

        // adding electrons to net86
        corrected_BE[86] = DEVICE_ACCESS(BE).back() + correction;

        // electron energy corrections
        corrected_BE[constants::proton] += -Eneutr;
        corrected_BE[constants::neutron] += -Eaneutr;

        /******************************************************/
        /* start computing reaction rate and their derivative */
        /******************************************************/

        int idx = 157 - 1 + 157 - 4 - 1, jdx = 157 - 1 + 157 - 4 - 1;
        // electron capture rates
        rates[++idx]  = rhoElec == 0 ? 0 : effe / rhoElec;
        drates[++jdx] = rhoElec == 0 ? 0 : deffe / rhoElec;

        rates[++idx]  = rhoElec == 0 ? 0 : effp / rhoElec;
        drates[++jdx] = rhoElec == 0 ? 0 : deffp / rhoElec;
    }
};

extern ComputeReactionRatesFunctor<double> computeReactionRates;
} // namespace nnet::net87