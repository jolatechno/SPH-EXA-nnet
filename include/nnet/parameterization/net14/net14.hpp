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
 * @brief net14 definition.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <vector>

#include "nnet_util/CUDA/cuda.inl"
#if COMPILE_DEVICE
#include "nnet_util/CUDA/cuda-util.hpp"
#endif

#include "net14_constants.hpp"

#include "nnet/nuclear_net.hpp"

#define BE_NET14                                                                                                       \
    28.2970 * constants::MevToErg COMMA 92.1631 * constants::MevToErg COMMA 127.621 *                                  \
        constants::MevToErg COMMA 160.652 * constants::MevToErg COMMA 198.259 * constants::MevToErg COMMA 236.539 *    \
        constants::MevToErg COMMA 271.784 * constants::MevToErg COMMA 306.719 * constants::MevToErg COMMA 342.056 *    \
        constants::MevToErg COMMA 375.479 * constants::MevToErg COMMA 411.470 * constants::MevToErg COMMA 447.704 *    \
        constants::MevToErg COMMA 483.995 * constants::MevToErg COMMA 526.850 * constants::MevToErg

namespace nnet::net14
{
/*! @brief if true print debuging prints. */
extern bool debug;

#ifdef NET14_NO_COULOMBIAN_DEBUG
/*! @brief if true ignore coulombian corrections. */
const bool skip_coulombian_correction = true;
#else
/*! @brief if true ignore coulombian corrections. */
const bool skip_coulombian_correction = false;
#endif

/*! @brief constant mass-excendent values */
DEVICE_DEFINE(inline static const std::array<double COMMA 14>, BE, = {BE_NET14};)

/*! @brief constant list of ordered reaction */
inline static const nnet::ReactionList reactionList(std::vector<nnet::Reaction>{
    /* !!!!!!!!!!!!!!!!!!!!!!!!
       3He -> C fusion */
    {{{0, 3}}, {{1}}},

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    C + He -> O fusion */
    {{{0}, {1}}, {{2}}},

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    O + He -> Ne fusion */
    {{{0}, {2}}, {{3}}},

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    fusions reactions from fits */
    {{{0}, {3}}, {{4}}},   // Ne + He -> Mg
    {{{0}, {4}}, {{5}}},   // Mg + He -> Si
    {{{0}, {5}}, {{6}}},   // Si + He -> S
    {{{0}, {6}}, {{7}}},   // S  + He -> Ar
    {{{0}, {7}}, {{8}}},   // Ar + He -> Ca
    {{{0}, {8}}, {{9}}},   // Ca + He -> Ti
    {{{0}, {9}}, {{10}}},  // Ti + He -> Cr
    {{{0}, {10}}, {{11}}}, // Cr + He -> Fe
    {{{0}, {11}}, {{12}}}, // Fe + He -> Ni
    {{{0}, {12}}, {{13}}}, // Ni + He -> Zn

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    2C -> Ne + He fusion
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{1, 2}}, {{3}, {0}}},

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    C + O -> Mg + He fusion
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{1}, {2}}, {{4}, {0}}},

    /* !!!!!!!!!!!!!!!!!!!!!!!!
    2O -> Si + He fusion
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{2, 2}}, {{5}, {0}}},

    /* 3He <- C fission
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{1}}, {{0, 3}}},

    /* C + He <- O fission
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{2}}, {{0}, {1}}},

    /* O + He <- Ne fission
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{3}}, {{0}, {2}}},

    /* fission reactions from fits
    !!!!!!!!!!!!!!!!!!!!!!!! */
    {{{4}}, {{0}, {3}}},   // Ne + He <- Mg
    {{{5}}, {{0}, {4}}},   // Mg + He <- Si
    {{{6}}, {{0}, {5}}},   // Si + He <- S
    {{{7}}, {{0}, {6}}},   // S  + He <- Ar
    {{{8}}, {{0}, {7}}},   // Ar + He <- Ca
    {{{9}}, {{0}, {8}}},   // Ca + He <- Ti
    {{{10}}, {{0}, {9}}},  // Ti + He <- Cr
    {{{11}}, {{0}, {10}}}, // Cr + He <- Fe
    {{{12}}, {{0}, {11}}}, // Fe + He <- Ni
    {{{13}}, {{0}, {12}}}  // Ni + He <- Zn
});

template<typename Float>
extern HOST_DEVICE_FUN void computeNet14ReactionRates(const Float* Y, const Float T, const Float rho,
                                                      const nnet::eos_struct<Float>& eos_struct, Float* corrected_BE,
                                                      Float* rates, Float* drates);

/*! @brief net14 functor */
template<typename Float>
class ComputeReactionRatesFunctor : public nnet::ComputeReactionRatesFunctor<Float>
{
public:
    ComputeReactionRatesFunctor() {}

    /*! @brief compute net14 rates
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
        computeNet14ReactionRates(Y, T, rho, eos_struct, corrected_BE, rates, drates);
    }
};

extern ComputeReactionRatesFunctor<double> computeReactionRates;

} // namespace nnet::net14