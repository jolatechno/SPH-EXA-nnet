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
 * @brief Interface definition for CUDA integration functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "nnet/nuclear_net.hpp"

namespace nnet::parallel
{
template<typename Float, class nseFunction = void*>
extern void computeNuclearReactionsImpl(const size_t n_particles, const int dimension, Float* rho_, Float* rho_m1_,
                                        Float** Y_, Float* temp_, Float* dt_, const Float hydro_dt,
                                        const Float previous_dt, const nnet::ReactionList& reactions,
                                        const nnet::ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                        const nnet::EosFunctor<Float>& eos, bool use_drhodt,
                                        const nseFunction jumpToNse = NULL);

template<typename Float>
extern void computeHelmholtzImpl(const size_t n_particles, const int dimension, const Float* Z, const Float* temp_,
                                 const Float* rho_, Float* const* Y_, Float* u, Float* cv, Float* p, Float* c,
                                 Float* dpdT);
} // namespace nnet::parallel