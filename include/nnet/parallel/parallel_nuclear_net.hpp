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

#pragma once

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

#include "implementation/parallel_nuclear_net_impl.hpp"

namespace nnet::parallel
{
/*! @brief function to compute nuclear reaction, either from NuclearData or ParticuleData if it includes Y
 *
 * @param n                   nuclearDataType including a field of nuclear abundances "Y"
 * @param firstIndex          first (included) particle considered in n
 * @param lastIndex           last (excluded) particle considered in n
 * @param hydro_dt            integration timestep
 * @param previous_dt         previous integration timestep
 * @param reactions           reaction list
 * @param construct_rates_BE  function constructing rates, rate derivatives and binding energies
 * @param eos                 equation of state
 * @param use_drhodt          if true considers drho/dt in eos
 * @param jumpToNse           function to jump to nuclear statistical equilibrium
 *
 * The minimum requierment for n (of type Data) are :
 *  - A field "Y" containing a vector of vector
 *  - Fields "temp", "rho" and "rho_m1"
 *  - "isAllocated()" function (see SPH-EXA)
 *  - "fieldNames" list of field names
 *  - A "AcceleratorType" template (that results in "true" from sphexa::HaveGpu<AcceleratorType> for GPU acceleration
 * and false for CPU computation).
 *
 * For the GPU version, Data should contain a GPU class n.devData with the exact same requierments.
 */
template<bool use_gpu, class Data, typename Float, class nseFunction = void*>
void computeNuclearReactions(Data& n, size_t firstIndex, size_t lastIndex, const Float hydro_dt,
                             const Float previous_dt, const nnet::ReactionList& reactions,
                             const nnet::ComputeReactionRatesFunctor<Float>& construct_rates_BE,
                             const nnet::EosFunctor<Float>& eos, bool use_drhodt, const nseFunction jumpToNse = NULL)
{
    const size_t n_particles = n.temp.size();
    const int    dimension   = n.numSpecies;

    if constexpr (use_gpu && COMPILE_DEVICE)
    {

        /* !!!!!!!!!!!!!
        GPU non-batch solver
        !!!!!!!!!!!!! */

#if COMPILE_DEVICE
        // check for drho/dt allocation
        Float* rho_m1_ptr = nullptr;
        if (use_drhodt)
        {
            int rho_m1_idx =
                std::distance(n.devData.fieldNames.begin(),
                              std::find(n.devData.fieldNames.begin(), n.devData.fieldNames.end(), "rho_m1"));
            if (!n.devData.isAllocated(rho_m1_idx))
            {
                use_drhodt = false;
                std::cerr << "disabeling using drho/dt because 'rho_m1' isn't alocated !\n";
            }
        }
        if (use_drhodt) rho_m1_ptr = (Float*)thrust::raw_pointer_cast(n.devData.rho_m1.data() + firstIndex);

        // reactions to GPU
        nnet::GPUReactionList dev_reactions = nnet::moveToGpu(reactions);

        // copy pointers to GPU
        std::vector<Float*> Y_raw_ptr(dimension);
        Float**             Y_dev_ptr;
        gpuErrchk(cudaMalloc((void**)&Y_dev_ptr, dimension * sizeof(Float*)));
        for (int i = 0; i < dimension; ++i)
            Y_raw_ptr[i] =
                (Float*)thrust::raw_pointer_cast(n.devData.Y[i].data() + firstIndex); // store Y raw pointer to CPU
        gpuErrchk(
            cudaMemcpy((void*)Y_dev_ptr, (void*)Y_raw_ptr.data(), dimension * sizeof(Float*), cudaMemcpyHostToDevice));

        // call the cuda kernel wrapper
        cudaComputeNuclearReactions(lastIndex - firstIndex, dimension, n.devData.buffer,
                                    (Float*)thrust::raw_pointer_cast(n.devData.rho.data() + firstIndex), rho_m1_ptr,
                                    Y_dev_ptr, (Float*)thrust::raw_pointer_cast(n.devData.temp.data() + firstIndex),
                                    (Float*)thrust::raw_pointer_cast(n.devData.dt.data() + firstIndex), hydro_dt,
                                    previous_dt, dev_reactions, construct_rates_BE, eos, use_drhodt);

        // free cuda buffer
        gpuErrchk(cudaFree((void*)Y_dev_ptr));
        free(dev_reactions);
#endif
    }
    else
    {

        /* !!!!!!!!!!!!!!!!!!!!!!!
        simple CPU parallel solver
        !!!!!!!!!!!!!!!!!!!!!!! */

        if (use_drhodt)
        {
            int rho_m1_idx =
                std::distance(n.fieldNames.begin(), std::find(n.fieldNames.begin(), n.fieldNames.end(), "rho_m1"));
            if (!n.isAllocated(rho_m1_idx))
            {
                use_drhodt = false;
                std::cerr << "disabeling using drho/dt because 'rho_m1' isn't alocated !\n";
            }
        }

        // copy pointers to array
        std::vector<Float*> Y_raw_ptr(dimension);
        for (int i = 0; i < dimension; ++i)
            Y_raw_ptr[i] = n.Y[i].data() + firstIndex;

        // call the cuda kernel wrapper
        computeNuclearReactionsImpl(lastIndex - firstIndex, dimension, n.rho.data() + firstIndex,
                                    n.rho_m1.data() + firstIndex, Y_raw_ptr.data(), n.temp.data() + firstIndex,
                                    n.dt.data() + firstIndex, hydro_dt, previous_dt, reactions, construct_rates_BE, eos,
                                    use_drhodt, jumpToNse);
    }
}

/*! @brief function to copute the helmholtz eos
 *
 * @param n           nuclearDataType including a field of nuclear abundances "Y"
 * @param firstIndex  first (included) particle considered in n
 * @param lastIndex   last (excluded) particle considered in n
 * @param Z           vector of number of charge (used in eos)
 *
 * The minimum requierment for n (of type Data) are :
 *  - A field "Y" containing a vector of vector
 *  - Fields "temp", "rho", "u", "cv", "p", "c" and "dpdT"
 *  - "isAllocated()"" function (see SPH-EXA)
 *  - "fieldNames" list of field names
 *  - A "AcceleratorType" template (that results in "true" from sphexa::HaveGpu<AcceleratorType> for GPU acceleration
 * and false for CPU computation).
 *
 * For the GPU version, Data should contain a GPU class n.devData with the exact same requierments.
 */
template<bool use_gpu, class Data, class Vector>
void computeHelmEOS(Data& n, size_t firstIndex, size_t lastIndex, const Vector& Z)
{
    const int dimension = n.numSpecies;
    using Float         = typename std::remove_reference<decltype(n.cv[0])>::type;

    if constexpr (use_gpu && COMPILE_DEVICE)
    {

        /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        simple GPU application of the eos
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#if COMPILE_DEVICE
        // copy data to the GPU
        Float* Z_dev;
        gpuErrchk(cudaMalloc((void**)&Z_dev, dimension * sizeof(Float)));
        gpuErrchk(cudaMemcpy((void*)Z_dev, (void*)Z.data(), dimension * sizeof(Float), cudaMemcpyHostToDevice));

        // copy pointers to GPU
        std::vector<Float*> Y_raw_ptr(dimension);
        Float**             Y_dev_ptr;
        gpuErrchk(cudaMalloc((void**)&Y_dev_ptr, dimension * sizeof(Float*)));
        for (int i = 0; i < dimension; ++i)
            Y_raw_ptr[i] =
                (Float*)thrust::raw_pointer_cast(n.devData.Y[i].data() + firstIndex); // store Y raw pointer to CPU
        gpuErrchk(
            cudaMemcpy((void*)Y_dev_ptr, (void*)Y_raw_ptr.data(), dimension * sizeof(Float*), cudaMemcpyHostToDevice));

        // call the cuda kernel wrapper
        cudaComputeHelmholtz(lastIndex - firstIndex, dimension, Z_dev,
                             // read buffers:
                             (Float*)thrust::raw_pointer_cast(n.devData.temp.data() + firstIndex),
                             (Float*)thrust::raw_pointer_cast(n.devData.rho.data() + firstIndex), Y_dev_ptr,
                             // write buffers:
                             (Float*)thrust::raw_pointer_cast(n.devData.u.data() + firstIndex),
                             (Float*)thrust::raw_pointer_cast(n.devData.cv.data() + firstIndex),
                             (Float*)thrust::raw_pointer_cast(n.devData.p.data() + firstIndex),
                             (Float*)thrust::raw_pointer_cast(n.devData.c.data() + firstIndex),
                             (Float*)thrust::raw_pointer_cast(n.devData.dpdT.data() + firstIndex));

        // debuging: check for error
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // free cuda buffer
        gpuErrchk(cudaFree((void*)Z_dev));
        gpuErrchk(cudaFree((void*)Y_dev_ptr));
#endif
    }
    else
    {

        /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        simple CPU parallel application of the eos
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

        // copy pointers to array
        std::vector<Float*> Y_raw_ptr(dimension);
        for (int i = 0; i < dimension; ++i)
            Y_raw_ptr[i] = n.Y[i].data() + firstIndex;

        // call the cpu impl
        computeHelmholtzImpl(lastIndex - firstIndex, dimension, Z.data(),
                             // read buffers:
                             n.temp.data() + firstIndex, n.rho.data() + firstIndex, Y_raw_ptr.data(),
                             // write buffers:
                             n.u.data() + firstIndex, n.cv.data() + firstIndex, n.p.data() + firstIndex,
                             n.c.data() + firstIndex, n.dpdT.data() + firstIndex);
    }
}
} // namespace nnet::parallel