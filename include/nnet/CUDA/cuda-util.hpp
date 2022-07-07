#pragma once

#include <cuda_runtime.h>

#ifdef USE_MPI
	#include <mpi.h>
#endif

void gpuErrchk(cudaError_t code) {
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
	/// function to move a buffer to the GPU
	/**
	 * TODO
	 */
	template<class T>
	T *move_to_gpu(const T* const ptr, int dimension) {
		T *dev_ptr;
		
		gpuErrchk(cudaMalloc((void**)&dev_ptr, dimension*sizeof(T)));
		gpuErrchk(cudaMemcpy(dev_ptr,     ptr, dimension*sizeof(T), cudaMemcpyHostToDevice));

		return dev_ptr;
	}

	/// function to free a from GPU
	/**
	 * TODO
	 */
	template<class T>
	void free_from_gpu(const T *dev_ptr) {
		gpuErrchk(cudaFree((void*)const_cast<T*>(dev_ptr)));
	}


#ifdef USE_MPI
	/// function to set a single CUDA device per local MPI rank
	/**
	 * TODO
	 */
	void initCudaMpi(MPI_Comm comm) {
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
