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
 * @brief net86 definition.
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
#include "nnet_util/algorithm.hpp"

#include "net86_constants.hpp"

#define BE_NET86                                                                                                       \
    0 COMMA 0 COMMA 28.2970 * constants::MevToErg COMMA 92.1631 * constants::MevToErg COMMA 127.621 *                  \
        constants::MevToErg COMMA 160.651 * constants::MevToErg COMMA 163.082 * constants::MevToErg COMMA 198.263 *    \
        constants::MevToErg COMMA 181.731 * constants::MevToErg COMMA 167.412 * constants::MevToErg COMMA 186.570 *    \
        constants::MevToErg COMMA 168.584 * constants::MevToErg COMMA 174.152 * constants::MevToErg COMMA 177.776 *    \
        constants::MevToErg COMMA 200.534 * constants::MevToErg COMMA 236.543 * constants::MevToErg COMMA 219.364 *    \
        constants::MevToErg COMMA 205.594 * constants::MevToErg COMMA 224.958 * constants::MevToErg COMMA 206.052 *    \
        constants::MevToErg COMMA 211.901 * constants::MevToErg COMMA 216.687 * constants::MevToErg COMMA 239.291 *    \
        constants::MevToErg COMMA 271.786 * constants::MevToErg COMMA 256.744 * constants::MevToErg COMMA 245.017 *    \
        constants::MevToErg COMMA 262.924 * constants::MevToErg COMMA 243.691 * constants::MevToErg COMMA 250.612 *    \
        constants::MevToErg COMMA 255.626 * constants::MevToErg COMMA 274.063 * constants::MevToErg COMMA 306.722 *    \
        constants::MevToErg COMMA 291.468 * constants::MevToErg COMMA 280.428 * constants::MevToErg COMMA 298.215 *    \
        constants::MevToErg COMMA 278.727 * constants::MevToErg COMMA 285.570 * constants::MevToErg COMMA 291.845 *    \
        constants::MevToErg COMMA 308.580 * constants::MevToErg COMMA 342.059 * constants::MevToErg COMMA 326.418 *    \
        constants::MevToErg COMMA 315.511 * constants::MevToErg COMMA 333.730 * constants::MevToErg COMMA 313.129 *    \
        constants::MevToErg COMMA 320.654 * constants::MevToErg COMMA 327.349 * constants::MevToErg COMMA 343.144 *    \
        constants::MevToErg COMMA 375.482 * constants::MevToErg COMMA 359.183 * constants::MevToErg COMMA 350.422 *    \
        constants::MevToErg COMMA 366.832 * constants::MevToErg COMMA 346.912 * constants::MevToErg COMMA 354.694 *    \
        constants::MevToErg COMMA 361.903 * constants::MevToErg COMMA 377.096 * constants::MevToErg COMMA 411.469 *    \
        constants::MevToErg COMMA 395.135 * constants::MevToErg COMMA 385.012 * constants::MevToErg COMMA 403.369 *    \
        constants::MevToErg COMMA 381.982 * constants::MevToErg COMMA 390.368 * constants::MevToErg COMMA 398.202 *    \
        constants::MevToErg COMMA 413.553 * constants::MevToErg COMMA 447.703 * constants::MevToErg COMMA 431.520 *    \
        constants::MevToErg COMMA 422.051 * constants::MevToErg COMMA 440.323 * constants::MevToErg COMMA 417.703 *    \
        constants::MevToErg COMMA 426.636 * constants::MevToErg COMMA 435.051 * constants::MevToErg COMMA 449.302 *    \
        constants::MevToErg COMMA 483.994 * constants::MevToErg COMMA 467.353 * constants::MevToErg COMMA 458.387 *    \
        constants::MevToErg COMMA 476.830 * constants::MevToErg COMMA 453.158 * constants::MevToErg COMMA 462.740 *    \
        constants::MevToErg COMMA 471.765 * constants::MevToErg COMMA 484.689 * constants::MevToErg COMMA 514.999 *    \
        constants::MevToErg COMMA 500.002 * constants::MevToErg COMMA 494.241 * constants::MevToErg COMMA 509.878 *    \
        constants::MevToErg COMMA 486.966 * constants::MevToErg COMMA 497.115 * constants::MevToErg COMMA 506.460 *    \
        constants::MevToErg

namespace nnet::net86
{
/*! @brief if true print debuging prints */
extern bool debug;

#ifdef NET86_NO_COULOMBIAN_DEBUG
/*! @brief if true ignore coulombian corrections */
const bool skip_coulombian_correction = true;
#else
/*! @briefif true ignore coulombian corrections */
const bool skip_coulombian_correction = false;
#endif

/*! @brief constant mass-excendent values */
DEVICE_DEFINE(inline static const std::array<double COMMA 86>, BE, = {BE_NET86};)

/*! @brief constant list of ordered reaction */
inline static const nnet::ReactionList reactionList = []()
{
    nnet::ReactionList reactions;

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    2C fusion,
    C + O fusion,
    2O fusion
    !!!!!!!!!!!!!!!!!!!!!!!! */
    reactions.pushBack(
        nnet::Reaction{{{constants::main_reactant[0], 2}}, {{constants::main_product[0]}, {constants::alpha}}});
    reactions.pushBack(nnet::Reaction{{{constants::main_reactant[1]}, {constants::secondary_reactant[1]}},
                                      {{constants::main_product[1]}, {constants::alpha}}});
    reactions.pushBack(
        nnet::Reaction{{{constants::main_reactant[2], 2}}, {{constants::main_product[2]}, {constants::alpha}}});

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    3He -> C fusion
    !!!!!!!!!!!!!!!!!!!!!!!! */
    reactions.pushBack(nnet::Reaction{{{constants::main_reactant[4], 3}}, {{constants::main_product[4]}}});

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    direct reaction
    !!!!!!!!!!!!!!!!!!!!!!!! */
    for (int i = 5; i < 157; ++i)
    {
        const int r1 = constants::main_reactant[i], r2 = constants::secondary_reactant[i],
                  p = constants::main_product[i];

        int delta_Z = constants::Z[r1] + constants::Z[r2] - constants::Z[p],
            delta_A = constants::A[r1] + constants::A[r2] - constants::A[p];

        if (delta_Z == 0 && delta_A == 0)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}},
                                              {{constants::main_product[i]}}});
        }
        else if (delta_A == 1 && delta_Z == 0)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}},
                                              {{constants::main_product[i]}, {constants::neutron}}});
        }
        else if (delta_A == 1 && delta_Z == 1)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}},
                                              {{constants::main_product[i]}, {constants::proton}}});
        }
        else if (delta_A == 4 && delta_Z == 2)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_reactant[i]}, {constants::secondary_reactant[i]}},
                                              {{constants::main_product[i]}, {constants::alpha}}});
        }
        else
            throw std::runtime_error("Mass conservation not possible when adding reaction to net86\n");
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    C -> 3He fission
    !!!!!!!!!!!!!!!!!!!!!!!! */
    reactions.pushBack(nnet::Reaction{{{constants::main_product[4]}}, {{constants::main_reactant[4], 3}}});

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    inverse reaction
    !!!!!!!!!!!!!!!!!!!!!!!! */
    for (int i = 5; i < 157; ++i)
    {
        const int r = constants::main_product[i], p1 = constants::main_reactant[i],
                  p2 = constants::secondary_reactant[i];

        int delta_Z = constants::Z[r] - (constants::Z[p1] + constants::Z[p2]),
            delta_A = constants::A[r] - (constants::A[p1] + constants::A[p2]);

        if (delta_Z == 0 && delta_A == 0)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_product[i]}},
                                              {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
        }
        else if (delta_A == -1 && delta_Z == 0)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_product[i]}, {constants::neutron}},
                                              {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
        }
        else if (delta_A == -1 && delta_Z == -1)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_product[i]}, {constants::proton}},
                                              {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
        }
        else if (delta_A == -4 && delta_Z == -2)
        {
            reactions.pushBack(nnet::Reaction{{{constants::main_product[i]}, {constants::alpha}},
                                              {{constants::main_reactant[i]}, {constants::secondary_reactant[i]}}});
        }
        else
            throw std::runtime_error("Mass conservation not possible when adding reaction to net86\n");
    }

    return reactions;
}();

/*! @brief compute net86 rates
 *
 * @param Y              molar fractions
 * @param T             temperature
 * @param rho           density
 * @param eos_struct    eos struct to populate
 * @param corrected_BE  will be populated by binding energies, corrected by coulombien terms
 * @param rates         will be populated with reaction rates
 * @param drates        will be populated with the temperature derivatives of reaction rates
 */
template<typename Float>
extern HOST_DEVICE_FUN void computeNet86ReactionRates(const Float* Y, const Float T, const Float rho,
                                                      const nnet::eos_struct<Float>& eos_struct, Float* corrected_BE,
                                                      Float* rates, Float* drates);

/*! @brief net86 functor */
template<typename Float>
class ComputeReactionRatesFunctor : public nnet::ComputeReactionRatesFunctor<Float>
{
public:
    ComputeReactionRatesFunctor() {}

    /*! @brief compute net86 rates
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
        computeNet86ReactionRates(Y, T, rho, eos_struct, corrected_BE, rates, drates);
    }
};

extern ComputeReactionRatesFunctor<double> computeReactionRates;
} // namespace nnet::net86