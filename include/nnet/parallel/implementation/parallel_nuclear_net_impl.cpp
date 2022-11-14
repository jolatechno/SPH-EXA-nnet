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
 * @brief Parallel application of nuclear networks and helmholtz EOS
 *
 * Applied on a class similar to ../sphexa/nuclear-data.hpp. Minimum class requierments are described in function
 * description
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#include <numeric>
#include <omp.h>

#include "nnet_util/CUDA/cuda.inl"
//#if COMPILE_DEVICE
#if defined(USE_CUDA)
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "nnet/CUDA/nuclear_net.cuh"
#include "CUDA/parallel_nuclear_net.cuh"
#endif

#include "nnet/nuclear_net.hpp"

#include "nnet_util/eigen.hpp"
#include "nnet_util/algorithm.hpp"

#include "nnet/parameterization/eos/helmholtz.hpp"

#include "parallel_nuclear_net_impl.hpp"

namespace nnet::parallel
{

template<typename Float, class nseFunction>
void computeNuclearReactionsImpl(const size_t n_particles, const int dimension, Float* rho_, Float* rho_m1_, Float** Y_,
                                 Float* temp_, Float* dt_, const Float hydro_dt, const Float previous_dt,
                                 const nnet::ReactionList&                       reactions,
                                 const nnet::ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                                 const nnet::EosFunctor<Float>& eos, bool use_drhodt, const nseFunction jumpToNse)
{
    // buffers
    std::vector<Float>   rates(reactions.size());
    eigen::Vector<Float> RHS(dimension + 1), DY_T(dimension + 1), Y_buffer(dimension), Y(dimension);
    eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

    int num_threads;
#pragma omp parallel
#pragma omp master
    num_threads        = omp_get_num_threads();
    int omp_batch_size = ::util::dynamicBatchSize(n_particles, num_threads);

#pragma omp parallel for firstprivate(Y, Mp, RHS, DY_T, rates, Y_buffer) schedule(dynamic, omp_batch_size)
    for (size_t i = 0; i < n_particles; ++i)
        if (rho_[i] > nnet::constants::minRho && temp_[i] > nnet::constants::minTemp)
        {
            // copy to local vector
            for (int j = 0; j < dimension; ++j)
                Y[j] = Y_[j][i];

            // compute drho/dt
            Float drho_dt = 0;
            if (use_drhodt)
                if (rho_m1_[i] != 0) rho_m1_[i] = (rho_[i] - rho_m1_[i]) / previous_dt;

            // solve
            nnet::solveSystemSubstep(dimension, Mp.data(), RHS.data(), DY_T.data(), rates.data(), reactions,
                                     construct_rates_BE, eos, Y.data(), temp_[i], Y_buffer.data(), rho_[i], drho_dt,
                                     hydro_dt, dt_[i], jumpToNse);

            // copy from local vector
            for (int j = 0; j < dimension; ++j)
                Y_[j][i] = Y[j];
        }
}

template void computeNuclearReactionsImpl(const size_t n_particles, const int dimension, float* rho_, float* rho_m1_,
                                          float** Y_, float* temp_, float* dt_, const float hydro_dt,
                                          const float previous_dt, const nnet::ReactionList& reactions,
                                          const nnet::ComputeReactionRatesFunctor<float>& construct_rates_BE,
                                          const nnet::EosFunctor<float>& eos, bool use_drhodt, void* jumpToNse);
template void computeNuclearReactionsImpl(const size_t n_particles, const int dimension, double* rho_, double* rho_m1_,
                                          double** Y_, double* temp_, double* dt_, const double hydro_dt,
                                          const double previous_dt, const nnet::ReactionList& reactions,
                                          const nnet::ComputeReactionRatesFunctor<double>& construct_rates_BE,
                                          const nnet::EosFunctor<double>& eos, bool use_drhodt, void* jumpToNse);

template<typename Float>
void computeHelmholtzImpl(const size_t n_particles, const int dimension, const Float* Z, const Float* temp_,
                          const Float* rho_, Float* const* Y_, Float* u, Float* cv, Float* p, Float* c, Float* dpdT)
{
    std::vector<Float> Y(dimension);

#pragma omp parallel for firstprivate(Y) schedule(static)
    for (size_t i = 0; i < n_particles; ++i)
    {
        // copy to local vector
        for (int j = 0; j < dimension; ++j)
            Y[j] = Y_[j][i];

        // compute abar and zbar
        double abar = std::accumulate(Y.data(), Y.data() + dimension, (double)0.);
        double zbar = eigen::dot(Y.data(), Y.data() + dimension, Z);

        auto eos_struct = nnet::eos::helmholtz::helmholtzEos(abar, zbar, temp_[i], rho_[i]);

        u[i]    = eos_struct.u;
        cv[i]   = eos_struct.cv;
        p[i]    = eos_struct.p;
        c[i]    = eos_struct.c;
        dpdT[i] = eos_struct.dpdT;
    }
}

template void computeHelmholtzImpl(const size_t n_particles, const int dimension, const float* Z, const float* temp_,
                                   const float* rho_, float* const* Y_, float* u, float* cv, float* p, float* c,
                                   float* dpdT);
template void computeHelmholtzImpl(const size_t n_particles, const int dimension, const double* Z, const double* temp_,
                                   const double* rho_, double* const* Y_, double* u, double* cv, double* p, double* c,
                                   double* dpdT);
} // namespace nnet::parallel