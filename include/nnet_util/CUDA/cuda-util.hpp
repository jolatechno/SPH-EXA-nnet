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

#ifdef USE_MPI
	#include <mpi.h>
#endif

#include <cuda_runtime.h>
#include <iostream>

/*! @brief manage CUDA error.
 *
 * @param code    CUDA error code.
 */
void inline gpuErrchk(cudaError_t code) {
	if (code != cudaSuccess) {
#ifdef CUDA_ERROR_FATAL
		std::string err = "CUDA error (fatal) ! \"";
		err += cudaGetErrorString(code);
		err += "\"\n";
		
		throw std::runtime_error(err);
#else
		std::cerr << "\tCUDA error (non-fatal) ! \"" << cudaGetErrorString(code) << "\"\n";
#endif
	}
}

namespace cuda_util {
	/*! @brief Function to move buffer to device
	 * 
	 * @param ptr        buffer to copy
	 * @param dimension  buffer size to copy
	 */
	template<class T>
	T inline *move_to_gpu(const T* const ptr, int dimension) {
		T *dev_ptr;
		
		gpuErrchk(cudaMalloc((void**)&dev_ptr, dimension*sizeof(T)));
		gpuErrchk(cudaMemcpy(dev_ptr,     ptr, dimension*sizeof(T), cudaMemcpyHostToDevice));

		return dev_ptr;
	}

	
	/*! @brief Function to free buffer from device
	 * 
	 * @param dev_ptr  device buffer to free
	 */
	template<class T>
	void inline free_from_gpu(const T *dev_ptr) {
		gpuErrchk(cudaFree((void*)const_cast<T*>(dev_ptr)));
	}


#ifdef USE_MPI
	/*! @brief function to set a single CUDA device per local MPI rank
	 * 
	 * @param comm  MPI communicator to split into local ranks with a single different assigned GPU each.
	 */
	void inline initCudaMpi(MPI_Comm comm) {
		MPI_Comm localComm;
		int rank, local_rank, local_size;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_rank(localComm, &local_rank);
		MPI_Comm_size(localComm, &local_size);

		// assume one gpu per node
		gpuErrchk(cudaSetDevice(local_rank));
	}
#endif
}