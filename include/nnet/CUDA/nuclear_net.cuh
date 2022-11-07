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
 * @brief CUDA utility functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "nnet/nuclear_net.hpp"

#include "nnet_util/CUDA/cuda-util.hpp"

namespace nnet
{
/*! @brief Class for reaction list on GPU. */
class GPUReactionList : public PtrReactionList
{
private:
    friend GPUReactionList moveToGpu(const PtrReactionList& reactions);
    friend void inline free(GPUReactionList& reactions);

public:
    GPUReactionList() {}
    ~GPUReactionList(){};
};

/*! @brief copy CPU reaction list to GPU.
 *
 * @param reactions CPU reaction list to copy to GPU.
 *
 * Returns a GPU reaction list copied from CPU.
 */
GPUReactionList inline moveToGpu(const PtrReactionList& reactions)
{
    GPUReactionList dev_reactions;
    dev_reactions.numReactions = reactions.numReactions;

    dev_reactions.reactantProduct = cuda_util::moveToGpu<Reaction::ReactantProduct>(
        reactions.reactantProduct, reactions.reactantBegin[reactions.numReactions]);
    dev_reactions.reactantBegin = cuda_util::moveToGpu<int>(reactions.reactantBegin, reactions.numReactions + 1);
    dev_reactions.productBegin  = cuda_util::moveToGpu<int>(reactions.productBegin, reactions.numReactions);

    return dev_reactions;
}

/*! @brief free GPU reaction list from GPU memory.
 *
 * @param reactions GPU reaction lists
 */
void inline free(GPUReactionList& reactions)
{
    gpuErrchk(cudaFree((void*)reactions.reactantProduct));
    gpuErrchk(cudaFree((void*)reactions.reactantBegin));
    gpuErrchk(cudaFree((void*)reactions.productBegin));
}
} // namespace nnet