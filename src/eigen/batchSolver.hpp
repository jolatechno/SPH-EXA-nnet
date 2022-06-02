#include "eigen.hpp"

#if defined(USE_CUDA) && !defined(CPU_BATCH_SOLVER)

/**************************************************************************************************************************************/
/* mostly gotten from https://github.com/OrangeOwlSolutions/CUDA-Utilities/blob/70343897abbf7a5608a6739759437f44933a5fc6/Utilities.cu */
/*              and https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda                          */
/* compile:  nvcc -Xcompiler "-fopenmp -pthread -I/cm/shared/modules/generic/mpi/openmpi/4.0.1/include -pthread -L/cm/shared/modules/generic/mpi/openmpi/4.0.1/lib -lmpi -std=c++17" -DUSE_CUDA -lcublas -lcuda -lcudart -DUSE_MPI -DNOT_FROM_SPHEXA hydro-mockup.cpp -o hydro-mockup.out
/* launch:   mpirun --bind-to core --oversubscribe --map-by ppr:1:numa -x OMP_NUM_THREADS=6 hydro-mockup.out -n 2 --test-case C-O-burning --n-particle 100
/**************************************************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace eigen::batchSolver {
	/// batch size for the GPU batched solver
	size_t batch_size = 10000;

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
		    	const int pivot = pivotArray[i] - 1;
		    	std::swap(vec[i], vec[pivot]);
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
		Float *dev_vec_Buffer, *dev_mat_Buffer, **dev_mat_ptr, **dev_vec_ptr;
		int *dev_pivotArray=NULL, *dev_InfoArray;

		// CPU Clusters
		std::vector<int> pivotArray, InfoArray;
		std::vector<Float*> mat_ptr, vec_ptr;
		std::vector<Float> vec_Buffer;
		std::vector<Float> mat_Buffer;
		
		// handle
		cublasHandle_t cublas_handle;
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		batch_solver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
			static_assert(std::is_same<Float, double>::value, "type in CUDA batch_solver must be DOUBLE for now");

			// --- CUBLAS initialization
		    util::cublasSafeCall(cublasCreate(&cublas_handle));

			/* debug: */
			int deviceCount;
		    cudaError_t e = cudaGetDeviceCount(&deviceCount);
		    std::cout << "(batch_solver) number of cuda device:" << (e == cudaSuccess ? deviceCount : -1) << "\n";

		    // alloc CPU buffers
		    vec_Buffer.resize(size*dimension);
			mat_Buffer.resize(size*dimension*dimension);
			InfoArray.resize(size);
#ifdef PIVOTING_IMPLEMENTED
			pivotArray.resize(size*dimension);
#endif
			mat_ptr.resize(size);
			vec_ptr.resize(size);

			// alloc GPU buffers
			util::gpuErrchk(cudaMalloc((void**)&dev_vec_Buffer,           dimension*size*sizeof(Float)));
			util::gpuErrchk(cudaMalloc((void**)&dev_mat_Buffer, dimension*dimension*size*sizeof(Float)));
#ifdef PIVOTING_IMPLEMENTED
			util::gpuErrchk(cudaMalloc((void**)&dev_pivotArray,           dimension*size*sizeof(int)));
#endif
			util::gpuErrchk(cudaMalloc((void**)&dev_InfoArray,                      size*sizeof(int)));
			util::gpuErrchk(cudaMalloc((void**)&dev_mat_ptr,                        size*sizeof(double*)));
			util::gpuErrchk(cudaMalloc((void**)&dev_vec_ptr,                        size*sizeof(double*)));
		}

		~batch_solver() {
			util::gpuErrchk(cudaFree(dev_vec_Buffer));
			util::gpuErrchk(cudaFree(dev_mat_Buffer));
#ifdef PIVOTING_IMPLEMENTED
			util::gpuErrchk(cudaFree(dev_pivotArray));
#endif
			util::gpuErrchk(cudaFree(dev_InfoArray));
			util::gpuErrchk(cudaFree(dev_mat_ptr));
			util::gpuErrchk(cudaFree(dev_vec_ptr));

			util::cublasSafeCall(cublasDestroy(cublas_handle));
		}



		/// insert system into batch solver
		/**
		 * TODO
		 */
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			for (int j = 0; j < dimension; ++j)
				for (int k = 0; k < dimension; ++k)
					mat_Buffer[dimension*dimension*i + j + dimension*k] = M[j + dimension*k];
					//mat_Buffer[dimension*dimension*i + dimension*j + k] = M[j + dimension*k];

			for (int j = 0; j < dimension; ++j)
				vec_Buffer[dimension*i + j] = RHS[j];
		}



		/// solve systems
		/**
		 * TODO
		 */
		void solve(size_t n_solve) {
			if (n_solve > 0) {
				// --- Creating the array of pointers needed as input/output to the batched getrf
				for (int i = 0; i < n_solve; i++) {
					mat_ptr[i] = dev_mat_Buffer + dimension*dimension*i;
					vec_ptr[i] = dev_vec_Buffer + dimension*i;
				}
	    		util::gpuErrchk(cudaMemcpy(dev_mat_ptr, mat_ptr.data(), n_solve*sizeof(double*), cudaMemcpyHostToDevice));
	    		util::gpuErrchk(cudaMemcpy(dev_vec_ptr, vec_ptr.data(), n_solve*sizeof(double*), cudaMemcpyHostToDevice));

				// push memory to device
				util::gpuErrchk(cudaMemcpy(dev_mat_Buffer, mat_Buffer.data(), dimension*dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice));

				// LU decomposition
				util::cublasSafeCall(cublasDgetrfBatched(cublas_handle,
					dimension, dev_mat_ptr,
					dimension, dev_pivotArray,
					dev_InfoArray, n_solve));

				// get Info from device
				util::gpuErrchk(cudaMemcpy(InfoArray.data(), dev_InfoArray, n_solve*sizeof(int), cudaMemcpyDeviceToHost));
				// check for error in each matrix
				#pragma omp parallel for schedule(static)
				for (int i = 0; i < n_solve; ++i)
			        if (InfoArray[i] != 0) {
			        	std::string error = "Factorization of matrix " + std::to_string(i);
			        	error += " Failed: Matrix may be singular (error code=" + std::to_string(InfoArray[i]) += ")";

			            cudaDeviceReset();
			            throw std::runtime_error(error);
			        }

#ifdef PIVOTING_IMPLEMENTED
				// get pivot from device
				util::gpuErrchk(cudaMemcpy(pivotArray.data(), dev_pivotArray, dimension*n_solve*sizeof(int), cudaMemcpyDeviceToHost));
				// rearange
				util::rearrange(vec_Buffer.data(), pivotArray.data(), dimension*n_solve);
#endif
				// push vector data to device
				util::gpuErrchk(cudaMemcpy(dev_vec_Buffer, vec_Buffer.data(), dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice));

	    		const double alpha = 1.;
				// solve lower triangular part
	    		//util::cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,     dimension, 1, &alpha, dev_mat_Buffer, dimension, dev_vec_Buffer, n_solve));
	    		util::cublasSafeCall(cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
	    			dimension, 1, &alpha, dev_mat_ptr,
	    			dimension, dev_vec_ptr,
	    			dimension, n_solve));
	    		// solve upper triangular part
	    		//util::cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, dimension, 1, &alpha, dev_mat_Buffer, dimension, dev_vec_Buffer, n_solve));
	    		util::cublasSafeCall(cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	    			dimension, 1, &alpha, dev_mat_ptr,
	    			dimension, dev_vec_ptr,
	    			dimension, n_solve));

				// get memory from device
				util::gpuErrchk(cudaMemcpy(vec_Buffer.data(), dev_vec_Buffer, dimension*n_solve*sizeof(Float), cudaMemcpyDeviceToHost));
			}
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

namespace eigen::batchSolver {
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
			if (n_solve > 0) {
				#pragma omp parallel for schedule(dynamic)
				for (size_t i = 0; i < n_solve; ++i)
					res_Buffer[i] = eigen::solve(mat_Buffer[i], RHS_Buffer[i]);
			}
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