#include "eigen.hpp"

#if defined(USE_CUDA) && !defined(CPU_BATCH_SOLVER)


/**************************************************************************************************************************************/
/* mostly gotten from https://github.com/OrangeOwlSolutions/CUDA-Utilities/blob/70343897abbf7a5608a6739759437f44933a5fc6/Utilities.cu */
/*              and https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda                          */
/* compile:  nvcc -Xcompiler "-fopenmp -I/cm/shared/modules/generic/mpi/openmpi/4.0.1/include -pthread -L/cm/shared/modules/generic/mpi/openmpi/4.0.1/lib -lmpi -std=c++17" -lcublas -lcuda -lcudart -DUSE_MPI -DNOT_FROM_SPHEXA -DUSE_CUDA -DCUDA_DEBUG hydro-mockup.cpp -o hydro-mockup.out
/* launch:   OMP_NUM_THREADS=32 mpirun --map-by node:PE=32:span -n 1 hydro-mockup.out -n 10 --test-case C-O-burning --n-particle 1000
/**************************************************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <omp.h>

#ifdef USE_MPI
	#include <mpi.h>
#endif

#ifndef MAX_BATCH_SIZE
	#define MAX_BATCH_SIZE 200000
#endif
#ifndef MIN_BATCH_SIZE
	#define MIN_BATCH_SIZE 10000
#endif

namespace eigen::batchSolver {
	namespace constants {
		/// maximum batch size for the GPU batched solver
		size_t max_batch_size = MAX_BATCH_SIZE;
		/// minimum batch size before falling back to non-batcher CPU-solver
		size_t min_batch_size = MIN_BATCH_SIZE;
	}

	namespace util {
		/// function to check for CUDA errors
		/**
		 * TODO
		 */
		void cudaSafeCall(cudaError_t code) {
			if (code != cudaSuccess) {
				std::string error = "CUDA error: ";
				error += cudaGetErrorString(code);

				throw std::runtime_error(error);
			}
		}



		/// function to get the number of devices
		/**
		 * TODO
		 */
		int getNumDevice() {
			// get number of device
			int deviceCount;
		    cudaSafeCall(cudaGetDeviceCount(&deviceCount));
		    return deviceCount;
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
		void rearrange(Float *vec, int *pivotArray, int dimension, size_t size) {
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < size; ++i) {
				size_t begin = dimension*i;
			    for (int j = 0; j < dimension; j++) {
			    	const int pivot = pivotArray[begin + j] - 1;
			    	std::swap(vec[begin + j], vec[begin + pivot]);
			    }
			}
		}



		/// Cublas getrfBatched templated wrapper
		/**
		 * TODO
		 */
		template<typename Float>
		cublasStatus_t inline cublasGetrfBatched(
			cublasHandle_t handle,
			int n,
			Float *const Aarray[],
			int lda,
			int *PivotArray,
			int *infoArray,
			int batchSize) 
		{
			throw std::runtime_error("type not supported in cublasGetrfBatched !");
			return CUBLAS_STATUS_NOT_SUPPORTED;
		}
		cublasStatus_t inline cublasGetrfBatched(
			cublasHandle_t handle,
			int n,
			double *const Aarray[],
			int lda,
			int *PivotArray,
			int *infoArray,
			int batchSize) 
		{
			return cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
		}
		cublasStatus_t inline cublasGetrfBatched(
			cublasHandle_t handle,
			int n,
			float *const Aarray[],
			int lda,
			int *PivotArray,
			int *infoArray,
			int batchSize) 
		{
			return cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
		}


		/// cublas trsmBatched templated wrapper
		/**
		 * TODO
		 */
		template<typename Float>
		cublasStatus_t inline cublasTrsmBatched(
			cublasHandle_t    handle,
			cublasSideMode_t  side,
			cublasFillMode_t  uplo,
			cublasOperation_t trans,
			cublasDiagType_t  diag,
			int m,
			int n,
			const Float *alpha,
			const Float *const A[],
			int lda,
			Float *const B[],
			int ldb,
			int batchCount) 
		{
			throw std::runtime_error("type not supported in cublasTrsmBatched !");
			return CUBLAS_STATUS_NOT_SUPPORTED;
		}
		cublasStatus_t inline cublasTrsmBatched(
			cublasHandle_t    handle,
			cublasSideMode_t  side,
			cublasFillMode_t  uplo,
			cublasOperation_t trans,
			cublasDiagType_t  diag,
			int m,
			int n,
			const double *alpha,
			const double *const A[],
			int lda,
			double *const B[],
			int ldb,
			int batchCount) 
		{
			return cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
		}
		cublasStatus_t inline cublasTrsmBatched(
			cublasHandle_t    handle,
			cublasSideMode_t  side,
			cublasFillMode_t  uplo,
			cublasOperation_t trans,
			cublasDiagType_t  diag,
			int m,
			int n,
			const float *alpha,
			const float *const A[],
			int lda,
			float *const B[],
			int ldb,
			int batchCount) 
		{
			return cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
		}
	}



	/// batch GPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class batch_solver {
	private:
		int dimension;
		size_t size;

		// CPU Buffers
		std::vector<int> pivotArray, InfoArray;
		std::vector<Float*> mat_ptr, vec_ptr;
		std::vector<Float> vec_Buffer;
		std::vector<Float> mat_Buffer;

		// GPU Buffers
		std::vector<Float*> dev_vec_Buffer, dev_mat_Buffer;
		std::vector<Float**> dev_mat_ptr, dev_vec_ptr;
		std::vector<int*> dev_pivotArray, dev_InfoArray;
		int deviceCount;




		/// solve systems on spcefic device
		/**
		 * TODO
		 */
		void solve_on_device(size_t begin, size_t n_solve, int device) {
			// set device and stream
			cudaStream_t stream;
			util::cudaSafeCall(cudaSetDevice(device));
			util::cudaSafeCall(cudaStreamCreate(&stream));

			// set accordingly on cublas
			cublasHandle_t cublas_handle;
			util::cublasSafeCall(cublasCreate(&cublas_handle));
			util::cublasSafeCall(cublasSetStream(cublas_handle, stream));
			

			/***********************************/
			/* actually solving system in batch*/
			/***********************************/
			// cpu offsets
			const size_t vec_begin = dimension*begin;
			const size_t mat_begin = dimension*dimension*begin; 

			// gpu offsets
			const size_t gpu_begin = begin - n_solve*device/deviceCount;
			const size_t gpu_vec_begin = dimension*gpu_begin;
			const size_t gpu_mat_begin = dimension*dimension*gpu_begin; 

			// --- Creating the array of pointers needed as input/output to the batched getrf
			for (int i = 0; i < n_solve; i++) {
				mat_ptr[i + begin] = dev_mat_Buffer[device] + gpu_mat_begin + dimension*dimension*i;
				vec_ptr[i + begin] = dev_vec_Buffer[device] + gpu_vec_begin + dimension*i;
			}
    		util::cudaSafeCall(cudaMemcpyAsync(dev_mat_ptr[device] + gpu_begin, mat_ptr.data() + begin, n_solve*sizeof(Float*), cudaMemcpyHostToDevice, stream));
    		util::cudaSafeCall(cudaMemcpyAsync(dev_vec_ptr[device] + gpu_begin, vec_ptr.data() + begin, n_solve*sizeof(Float*), cudaMemcpyHostToDevice, stream));

			// push memory to device
			util::cudaSafeCall(cudaMemcpyAsync(dev_mat_Buffer[device] + gpu_mat_begin, mat_Buffer.data() + mat_begin, dimension*dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice, stream));

			// LU decomposition
			int* loc_dev_pivotArray = dev_pivotArray[device] == NULL ? NULL : dev_pivotArray[device] + gpu_vec_begin;
			util::cublasSafeCall(util::cublasGetrfBatched(cublas_handle,
				dimension, dev_mat_ptr[device]    + gpu_begin,
				dimension, loc_dev_pivotArray,
				dev_InfoArray[device] + gpu_begin, n_solve));

			// get Info from device
			util::cudaSafeCall(cudaMemcpyAsync(InfoArray.data() + begin, dev_InfoArray[device] + gpu_begin, n_solve*sizeof(int), cudaMemcpyDeviceToHost, stream));
			// check for error in each matrix
			for (int i = begin; i < n_solve + begin; ++i)
		        if (InfoArray[i] != 0) {
		        	std::string error = "Factorization of matrix " + std::to_string(i);
		        	error += " Failed: Matrix may be singular (error code=" + std::to_string(InfoArray[i]) += ")";

		            cudaDeviceReset();
		            throw std::runtime_error(error);
		        }


		    if (dev_pivotArray[device] != NULL) {
				// get pivot from device
				util::cudaSafeCall(cudaMemcpyAsync(pivotArray.data() + vec_begin, loc_dev_pivotArray, dimension*n_solve*sizeof(int), cudaMemcpyDeviceToHost, stream));
				// rearange
				util::rearrange(vec_Buffer.data() + vec_begin, pivotArray.data() + vec_begin, dimension, n_solve);
			}

			// push vector data to device
			util::cudaSafeCall(cudaMemcpyAsync(dev_vec_Buffer[device] + gpu_vec_begin, vec_Buffer.data() + vec_begin, dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice, stream));

    		const double alpha = 1.;
			// solve lower triangular part
    		util::cublasSafeCall(util::cublasTrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
    			dimension, 1, &alpha, dev_mat_ptr[device] + gpu_begin,
    			dimension,            dev_vec_ptr[device] + gpu_begin,
    			dimension, n_solve));
    		// solve upper triangular part
    		util::cublasSafeCall(util::cublasTrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
    			dimension, 1, &alpha, dev_mat_ptr[device] + gpu_begin,
    			dimension,            dev_vec_ptr[device] + gpu_begin,
    			dimension, n_solve));

			// get memory from device
			util::cudaSafeCall(cudaMemcpyAsync(vec_Buffer.data() + vec_begin, dev_vec_Buffer[device] + gpu_vec_begin, dimension*n_solve*sizeof(Float), cudaMemcpyDeviceToHost, stream));

			// destroy stream
			util::cudaSafeCall(cudaStreamDestroy(stream));
			// dehalocate handle
			util::cublasSafeCall(cublasDestroy(cublas_handle));
		}
		


		
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		batch_solver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
		    // alloc CPU buffers
		    vec_Buffer.resize(size*dimension);
			mat_Buffer.resize(size*dimension*dimension);
			InfoArray.resize(size);
			pivotArray.resize(size*dimension);
			mat_ptr.resize(size);
			vec_ptr.resize(size);

			// multi-GPU support here, one thread = one GPU
			// get number of device
			deviceCount = util::getNumDevice();
			// resize buffer
			dev_vec_Buffer.resize(deviceCount);
			dev_mat_Buffer.resize(deviceCount);
			dev_mat_ptr.resize(deviceCount);
			dev_vec_ptr.resize(deviceCount);
			dev_pivotArray.resize(deviceCount);
			dev_InfoArray.resize(deviceCount);
		    #pragma omp parallel num_threads(deviceCount)
		    {
		    	// get device id = thread id
		    	int device = omp_get_thread_num();

		    	// get range for solving
		    	size_t begin         = size*device/deviceCount;
		    	size_t end           = size*(device + 1)/deviceCount;
		    	size_t local_n_solve = end - begin;

		    	/*****************************/
				/* allocating device Buffers */
				/*****************************/

				// set device
				cudaSetDevice(device);

				// allocate GPU vectors
				util::cudaSafeCall(cudaMalloc((void**)&dev_vec_Buffer[device],           dimension*size*sizeof(Float)));
				util::cudaSafeCall(cudaMalloc((void**)&dev_mat_Buffer[device], dimension*dimension*size*sizeof(Float)));
				util::cudaSafeCall(cudaMalloc((void**)&dev_pivotArray[device],           dimension*size*sizeof(int)));
				util::cudaSafeCall(cudaMalloc((void**)&dev_InfoArray[device],                      size*sizeof(int)));
				util::cudaSafeCall(cudaMalloc((void**)&dev_mat_ptr[device],                        size*sizeof(Float*)));
				util::cudaSafeCall(cudaMalloc((void**)&dev_vec_ptr[device],                        size*sizeof(Float*)));
		    }
		}

		~batch_solver() {
		    // multi-GPU support here, one thread = one GPU
		    #pragma omp parallel num_threads(deviceCount)
		    {
		    	// get device id = thread id
		    	int device = omp_get_thread_num();

		    	/*******************************/
				/* deallocating device Buffers */
				/*******************************/

				// deallocate memory
				util::cudaSafeCall(cudaFree(dev_vec_Buffer[device]));
				util::cudaSafeCall(cudaFree(dev_mat_Buffer[device]));
				if (dev_pivotArray[device] != NULL)
					util::cudaSafeCall(cudaFree(dev_pivotArray[device]));
				util::cudaSafeCall(cudaFree(dev_InfoArray[device]));
				util::cudaSafeCall(cudaFree(dev_mat_ptr[device]));
				util::cudaSafeCall(cudaFree(dev_vec_ptr[device]));
		    }
		}



		/// get reference to system for insertion
		/**
		 * TODO
		 */
		std::tuple<Float*, Float*> get_system_reference(size_t i) {
			return {&mat_Buffer[dimension*dimension*i], &vec_Buffer[dimension*i]};
		}


		/// solve systems
		/**
		 * supports multi-GPU
		 */
		void solve(size_t n_solve) {
#ifdef CUDA_DEBUG
		    /* debug: */
			std::cout << "solving for " << n_solve << " particles on " << deviceCount << " devices\n";
#endif

		    // multi-GPU support here
		    #pragma omp parallel
		    {
		    	// thread id
		    	const int num_threads = omp_get_num_threads();
		    	const int thread_id = omp_get_thread_num();

		    	// get device id
		    	const int device_id = deviceCount*thread_id/num_threads;

		    	// get solving range for device
		    	const size_t device_begin =       device_id*n_solve/deviceCount;
		    	const size_t device_end   = (device_id + 1)*n_solve/deviceCount;
		    	const size_t device_size  = device_end - device_begin;
		    	// get thread range for device
		    	const size_t device_thread_begin =       device_id*num_threads/deviceCount;
		    	const size_t device_thread_end   = (device_id + 1)*num_threads/deviceCount;
		    	const size_t device_num_thread   = device_thread_end - device_thread_begin;
		    	const size_t device_thread_id    =         thread_id - device_thread_begin;

		    	// get range for solving
		    	size_t solve_begin = device_begin +       device_thread_id*device_size/device_num_thread;
		    	size_t solve_end   = device_begin + (device_thread_id + 1)*device_size/device_num_thread;
		    	size_t solve_size  = solve_end - solve_begin;

		    	// actually solve
		    	solve_on_device(solve_begin, solve_size, device_id);
		    }

#ifdef CUDA_DEBUG
		    /* debug: */
			std::cout << "\tfinished solving\n";
#endif
		}



		/// retrieve results
		/**
		 * TODO
		 */
		Float *get_res(size_t i) {
			return &vec_Buffer[dimension*i];
		}
	};
}

#else

/*******************************************************/
/* dummy CPU solver while awaiting a CUDA batch solver */
/*******************************************************/

#include <omp.h>
#include <vector>

#ifndef MAX_BATCH_SIZE
	#define MAX_BATCH_SIZE 1000
#endif
#ifndef MIN_BATCH_SIZE
	#define MIN_BATCH_SIZE 0
#endif

namespace eigen::batchSolver {
	namespace constants {
		/// maximum batch size for the CPU batched solver
		size_t max_batch_size = MAX_BATCH_SIZE;
		/// minimum batch size before falling back to non-batcher CPU-solver, equal to zero
		size_t min_batch_size = MIN_BATCH_SIZE;
	}

	namespace util {
		/// function to get the number of devices (=1)
		/**
		 * TODO
		 */
		int getNumDevice() {
			return 1;
		}
	}

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



		/// get reference to system for insertion
		/**
		 * TODO
		 */
		std::tuple<Float*, Float*> get_system_reference(size_t i) {
			return {mat_Buffer[i].data(), RHS_Buffer[i].data()};
		}



		/// solve systems
		/**
		 * TODO
		 */
		void solve(size_t n_solve) {
#ifdef CUDA_DEBUG
		    /* debug: */
			std::cout << "solving for " << n_solve << " particles on CPU\n";
#endif

			#pragma omp parallel for schedule(dynamic)
			for (size_t i = 0; i < n_solve; ++i)
				res_Buffer[i] = eigen::solve(mat_Buffer[i], RHS_Buffer[i]);


#ifdef CUDA_DEBUG
		    /* debug: */
			std::cout << "\tfinished solving\n";
#endif
		}



		/// retrieve results
		/**
		 * TODO
		 */
		auto &get_res(size_t i) {
			return res_Buffer[i];
		}
	};
}
#endif