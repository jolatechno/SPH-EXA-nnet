#include "eigen.hpp"

#ifndef CUDA_NOT_IMPLEMENTED

/**************************************************************************************************************************************/
/* mostly gotten from https://github.com/OrangeOwlSolutions/CUDA-Utilities/blob/70343897abbf7a5608a6739759437f44933a5fc6/Utilities.cu */
/*              and https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda                          */
/**************************************************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace eigen::cudasolver {
	/// batch size for the GPU batched solver
	size_t batch_size = 1000000;

	namespace util {
		/// function to check for CUDA errors
		/**
		 * TODO
		 */
		void gpuErrchk(cudaError_t code) {
			if (code != cudaSuccess) {
				std::string error = "CUDA error: ";
				error += cudaGetErrorString(code);

				throw std::runtime_error(error);
			}
		}
		


		/// function to check for cublas error
		/**
		 * TODO
		 */
		void cublasSafeCall(cublasStatus_t code) {
			if (code != CUBLAS_STATUS_SUCCESS) {
				std::string error = "CUBLAS error: ";
				error += std::to_string((int)code);

				throw std::runtime_error(error);
			}
		}



		/// function to rearange according to pivot
		/**
		 * TODO
		 */
		template<typename Float>
		void rearrange(Float *vec, int *pivotArray, int size) {
			#pragma omp parallel for schedule(static)
		    for (int i = 0; i < size; i++) {
		        double temp = vec[i];
		        vec[i] = vec[pivotArray[i] - 1];
		        vec[pivotArray[i] - 1] = temp;
		    }   
		}
	}



	/// batch GPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class batch_solver {
	private:
		size_t size;
		int dimension;

		// GPU Clusters
		Float *dev_vec_Buffer, *dev_mat_Buffer, **dev_inout_pointers;
		int *dev_pivotArray, *dev_InfoArray;

		// CPU Clusters
		std::vector<int> pivotArray, InfoArray;
		std::vector<Float*> inout_pointers;
		std::vector<Float> vec_Buffer;
		std::vector<Float> mat_Buffer;
	
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		batch_solver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
			/* debug: */
			int deviceCount;
		    cudaError_t e = cudaGetDeviceCount(&deviceCount);
		    std::cout << "(batch_solver) number of cuda device:" << (e == cudaSuccess ? deviceCount : -1) << "\n";

		    // alloc CPU buffers
		    vec_Buffer.resize(size*dimension);
			mat_Buffer.resize(size*dimension*dimension);
			InfoArray.resize(size);
			pivotArray.resize(size*dimension);

			// alloc GPU buffers
			util::gpuErrchk(cudaMalloc((void**)&dev_vec_Buffer,           dimension*size*sizeof(Float)));
			util::gpuErrchk(cudaMalloc((void**)&dev_mat_Buffer, dimension*dimension*size*sizeof(Float)));
			util::gpuErrchk(cudaMalloc((void**)&dev_pivotArray,           dimension*size*sizeof(int)));
			util::gpuErrchk(cudaMalloc((void**)&dev_InfoArray,                      size*sizeof(int)));
			util::gpuErrchk(cudaMalloc((void**)&dev_inout_pointers,                 size*sizeof(double*)));

			// --- Creating the array of pointers needed as input/output to the batched getrf
			for (int i = 0; i < size; i++)
				inout_pointers[i] = dev_mat_Buffer + dimension*dimension*i;
    		util::gpuErrchk(cudaMemcpy(dev_inout_pointers, inout_pointers.data(), size*sizeof(double*), cudaMemcpyHostToDevice));
		}



		/// insert system into batch solver
		/**
		 * TODO
		 */
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			for (int j = 0; j < dimension*dimension; ++j)
				mat_Buffer[dimension*dimension*i + j] = M[j]; 

			for (int j = 0; j < dimension; ++j)
				vec_Buffer[dimension*i + j] = M[j]; 
		}



		/// solve systems
		/**
		 * TODO
		 */
		void solve(size_t n_solve) {
			// --- CUBLAS initialization
		    cublasHandle_t cublas_handle;
		    util::cublasSafeCall(cublasCreate(&cublas_handle));

			// push memory to device
			util::gpuErrchk(cudaMemcpy(dev_mat_Buffer, mat_Buffer.data(), dimension*dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice));

			// LU decomposition
			util::cublasSafeCall(cublasDgetrfBatched(cublas_handle, n_solve, dev_inout_pointers, n_solve, dev_pivotArray, dev_InfoArray, n_solve));

			// get info and pivot from device
			util::gpuErrchk(cudaMemcpy(InfoArray.data(),  dev_InfoArray,            n_solve*sizeof(int), cudaMemcpyDeviceToHost));
			util::gpuErrchk(cudaMemcpy(pivotArray.data(), dev_pivotArray, dimension*n_solve*sizeof(int), cudaMemcpyDeviceToHost));
			// rearange
			util::rearrange(vec_Buffer.data(), pivotArray.data(), dimension*n_solve);
			// push vector data to device
			util::gpuErrchk(cudaMemcpy(dev_vec_Buffer, vec_Buffer.data(), dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice));

    		const double alpha = 1.;
			// solve lower triangular part
    		util::cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,     n_solve, 1, &alpha, dev_mat_Buffer, n_solve, dev_vec_Buffer, n_solve));
    		// solve upper triangular part
    		util::cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_solve, 1, &alpha, dev_mat_Buffer, n_solve, dev_vec_Buffer, n_solve));

			// get memory from device
			util::gpuErrchk(cudaMemcpy(vec_Buffer.data(), dev_vec_Buffer, dimension*n_solve*sizeof(Float), cudaMemcpyDeviceToHost));
		}



		/// retrieve results
		/**
		 * TODO
		 */
		void get_res(size_t i, Float *res) {
			for (int j = 0; j < dimension; ++j)
				res[j] = vec_Buffer[dimension*i + j];
		}
	};
}



#else

/*******************************************************/
/* dummy CPU solver while awaiting a CUDA batch solver */
/*******************************************************/

#include <omp.h>
#include <vector>

namespace eigen::cudasolver {
	/// batch size for the GPU batched solver
	size_t batch_size = 100;

	/// dummy batch CPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class batch_solver {
	private:
		int dimension;
		size_t size;
		std::vector<eigen::Vector<Float>> RHS_Buffer, res_Buffer;
		std::vector<eigen::Matrix<Float>> mat_Buffer;
	
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		batch_solver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
			RHS_Buffer.resize(size);
			res_Buffer.resize(size);
			mat_Buffer.resize(size);
			#pragma omp parallel for schedule(dynamic)
			for (size_t i = 0; i < size; ++i) {
				RHS_Buffer[i].resize(dimension);
				res_Buffer[i].resize(dimension);
				mat_Buffer[i].resize(dimension, dimension);
			}
		}



		/// insert system into batch solver
		/**
		 * TODO
		 */
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			for (int j = 0; j < dimension; ++j) {
				RHS_Buffer[i][j] = RHS[j];
				for (int k = 0; k < dimension; ++k)
					mat_Buffer[i](j, k) = M[j + dimension*k];
			}
		}



		/// solve systems
		/**
		 * TODO
		 */
		void solve(size_t n_solve) {
			#pragma omp parallel for schedule(dynamic)
			for (size_t i = 0; i < n_solve; ++i)
				res_Buffer[i] = eigen::solve(mat_Buffer[i], RHS_Buffer[i]);
		}



		/// retrieve results
		/**
		 * TODO
		 */
		void get_res(size_t i, Float *res) {
			for (int j = 0; j < dimension; ++j)
				res[j] = res_Buffer[i][j];
		}
	};
}
#endif