#include "eigen.hpp"

#include <omp.h>

#ifdef USE_MPI
	#include <mpi.h>
#endif

#ifdef USE_CUDA
	#include <cublas_v2.h>
	#include <cuda_runtime.h>
	#include <device_launch_parameters.h>
#endif

#ifndef MAX_BATCH_SIZE
	#define MAX_BATCH_SIZE 200000
#endif
#ifndef MIN_BATCH_SIZE
	#define MIN_BATCH_SIZE 10000
#endif

/******************/
/* base functions */
/******************/

namespace eigen::batchSolver {
	namespace constants {
		/// maximum batch size for the GPU batched solver
		size_t max_batch_size = MAX_BATCH_SIZE;
		/// minimum batch size before falling back to non-batcher CPU-solver
		size_t min_batch_size = MIN_BATCH_SIZE;

		/// first device id used
		int device_begin = 0;
		/// end of the device ranged used
		int device_end   = 1;
	}

	namespace util {
		/// set the first device to use
		/**
		 * TODO
		 */
		void set_device_range(int begin, int end) {
			constants::device_begin = begin;
			constants::device_end   = end;
		}

		/// get the number of devices used
		/**
		 * TODO
		 */
		int getNumDevice() {
			return constants::device_end - constants::device_begin;
		}
	}



#ifdef USE_CUDA

/**************************************************************************************************************************************/
/* mostly gotten from https://github.com/OrangeOwlSolutions/CUDA-Utilities/blob/70343897abbf7a5608a6739759437f44933a5fc6/Utilities.cu */
/*              and https://stackoverflow.com/questions/28794010/solving-dense-linear-systems-ax-b-with-cuda                          */
/* compile:  nvcc -Xcompiler "-fopenmp -I/cm/shared/modules/generic/mpi/openmpi/4.0.1/include -pthread -L/cm/shared/modules/generic/mpi/openmpi/4.0.1/lib -lmpi -std=c++17" -lcublas -lcuda -lcudart -DUSE_MPI -DNOT_FROM_SPHEXA -DCUDA_DEBUG_ -DCPU_BATCH_SOLVER_ -DUSE_CUDA hydro-mockup.cpp -o hydro-mockup.out
/* launch:   mpirun --oversubscribe --map-by node:PE=10:span -x OMP_NUM_THREADS=10 -n 2 hydro-mockup.out --test-case C-O-burning -n 10 --n-particle 50000 > res.out 2> err.out &
/**************************************************************************************************************************************/

	namespace cuda {
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
	}




	namespace cublas {
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




	namespace util {
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



#ifdef USE_MPI
		/// init device spread accross local ranks
		void MPI_init_device(MPI_Comm comm) {
			int rank, local_rank, local_size;
			MPI_Comm local_comm;
			MPI_Comm_rank(comm, &rank);

			MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank,  MPI_INFO_NULL, &local_comm);
			MPI_Comm_size(local_comm, &local_size);
			MPI_Comm_rank(local_comm, &local_rank);

			int num_device   = cuda::getNumDevice();
			int device_begin = local_rank*num_device/local_size;
			int device_end   = std::max((local_rank + 1)*num_device/local_size,
										device_begin + 1);
			util::set_device_range(device_begin, device_end);

#ifdef CUDA_DEBUG
			std::cout << "using device " << device_begin << " to " << device_end << " on rank " << local_rank << "/" << local_size << "\n";
#endif
		}
#endif
	}



	/// batch GPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class CUDAsolver {
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




		/// solve systems on spcefic device
		/**
		 * TODO
		 */
		void solve_on_device(size_t begin, size_t n_solve, int device) {
			// set device and stream
			cudaStream_t cublasStream;
			cuda::cudaSafeCall(cudaSetDevice(device));
			cuda::cudaSafeCall(cudaStreamCreate(&cublasStream));

			// set accordingly on cublas
			cublasHandle_t cublas_handle;
			cublas::cublasSafeCall(cublasCreate(&cublas_handle));
			cublas::cublasSafeCall(cublasSetStream(cublas_handle, cublasStream));
			

			/***********************************/
			/* actually solving system in batch*/
			/***********************************/
			// cpu offsets
			const size_t vec_begin = dimension*begin;
			const size_t mat_begin = dimension*dimension*begin; 

			// gpu offsets
			const int deviceCount = util::getNumDevice();
			const int device_id_  = device - constants::device_begin;
			const size_t gpu_begin = begin - n_solve*device_id_/deviceCount;
			const size_t gpu_vec_begin = dimension*gpu_begin;
			const size_t gpu_mat_begin = dimension*dimension*gpu_begin; 

			// push memory to device
			cuda::cudaSafeCall(cudaMemcpyAsync(dev_mat_Buffer[device_id_] + gpu_mat_begin, mat_Buffer.data() + mat_begin, dimension*dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice, cublasStream));

			// LU decomposition
			cublas::cublasSafeCall(cublas::cublasGetrfBatched(cublas_handle,
				dimension, dev_mat_ptr[device_id_]    + gpu_begin,
				dimension, dev_pivotArray[device_id_] + gpu_begin,
				dev_InfoArray[device_id_]             + gpu_begin, n_solve));

			// synchronize
			cuda::cudaSafeCall(cudaStreamSynchronize(cublasStream));
			// get pivot from device
			cuda::cudaSafeCall(cudaMemcpyAsync(pivotArray.data() + vec_begin, dev_pivotArray[device_id_] + gpu_begin, dimension*n_solve*sizeof(int), cudaMemcpyDeviceToHost, cublasStream));


#ifdef GPU_CHECK_ERRORS
			/* for safety (debug) */
			// get Info from device
			cuda::cudaSafeCall(cudaMemcpy(InfoArray.data() + begin, dev_InfoArray[device_id_] + gpu_begin, n_solve*sizeof(int), cudaMemcpyDeviceToHost));
			// check for error in each matrix
			for (int i = begin; i < n_solve + begin; ++i)
		        if (InfoArray[i] != 0) {
		        	std::string error = "Factorization of matrix " + std::to_string(i);
		        	error += " Failed: Matrix may be singular (error code=" + std::to_string(InfoArray[i]) += ")";

		            cudaDeviceReset();
		            throw std::runtime_error(error);
		        }
#endif


			// synchronize
			cuda::cudaSafeCall(cudaStreamSynchronize(cublasStream));
			// rearange
			util::rearrange(vec_Buffer.data() + vec_begin, pivotArray.data() + vec_begin, dimension, n_solve);
			// push vector data to device
			cuda::cudaSafeCall(cudaMemcpyAsync(dev_vec_Buffer[device_id_] + gpu_vec_begin, vec_Buffer.data() + vec_begin, dimension*n_solve*sizeof(Float), cudaMemcpyHostToDevice, cublasStream));

    		const double alpha = 1.;
			// solve lower triangular part
    		cublas::cublasSafeCall(cublas::cublasTrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
    			dimension, 1, &alpha, dev_mat_ptr[device_id_] + gpu_begin,
    			dimension,            dev_vec_ptr[device_id_] + gpu_begin,
    			dimension, n_solve));
    		// solve upper triangular part
    		cublas::cublasSafeCall(cublas::cublasTrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
    			dimension, 1, &alpha, dev_mat_ptr[device_id_] + gpu_begin,
    			dimension,            dev_vec_ptr[device_id_] + gpu_begin,
    			dimension, n_solve));

			// get memory from device
			cuda::cudaSafeCall(cudaMemcpyAsync(vec_Buffer.data() + vec_begin, dev_vec_Buffer[device_id_] + gpu_vec_begin, dimension*n_solve*sizeof(Float), cudaMemcpyDeviceToHost, cublasStream));

			// destroy stream
			cuda::cudaSafeCall(cudaStreamDestroy(cublasStream));
			// dehalocate handle
			cublas::cublasSafeCall(cublasDestroy(cublas_handle));
		}
		


		
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		CUDAsolver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
		    // alloc CPU buffers
		    vec_Buffer.resize(size*dimension);
			mat_Buffer.resize(size*dimension*dimension);
			InfoArray.resize(size);
			pivotArray.resize(size*dimension);
			mat_ptr.resize(size);
			vec_ptr.resize(size);

			// multi-GPU support here, one thread = one GPU
			// get number of device
			int deviceCount = util::getNumDevice();
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
		    	int device_id_ = omp_get_thread_num();
		    	int device     = constants::device_begin + device_id_;

		    	// get range for solving
		    	size_t begin         =       size*device_id_/deviceCount;
		    	size_t end           = size*(device_id_ + 1)/deviceCount;
		    	size_t local_size = end - begin;

		    	/*****************************/
				/* allocating device Buffers */
				/*****************************/

				// set device and stream
				cudaStream_t stream1, stream2;
				cuda::cudaSafeCall(cudaSetDevice(device));
				cuda::cudaSafeCall(cudaStreamCreate(&stream1));
				cuda::cudaSafeCall(cudaStreamCreate(&stream2));

				// allocate GPU vectors
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_vec_Buffer[device_id_],           dimension*local_size*sizeof(Float)));
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_mat_Buffer[device_id_], dimension*dimension*local_size*sizeof(Float)));
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_pivotArray[device_id_],           dimension*local_size*sizeof(int)));
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_InfoArray[device_id_],                      local_size*sizeof(int)));
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_mat_ptr[device_id_],                        local_size*sizeof(Float*)));
				cuda::cudaSafeCall(cudaMalloc((void**)&dev_vec_ptr[device_id_],                        local_size*sizeof(Float*)));

				// --- Creating the array of pointers needed as input/output to the batched getrf
				for (int i = 0; i < local_size; i++) {
					mat_ptr[i + begin] = dev_mat_Buffer[device_id_] + dimension*dimension*i;
					vec_ptr[i + begin] = dev_vec_Buffer[device_id_] + dimension*i;
				}
	    		cuda::cudaSafeCall(cudaMemcpyAsync(dev_mat_ptr[device_id_], mat_ptr.data() + begin, local_size*sizeof(Float*), cudaMemcpyHostToDevice, stream1));
	    		cuda::cudaSafeCall(cudaMemcpyAsync(dev_vec_ptr[device_id_], vec_ptr.data() + begin, local_size*sizeof(Float*), cudaMemcpyHostToDevice, stream2));

	    		// destroy stream
				cuda::cudaSafeCall(cudaStreamDestroy(stream1));
				cuda::cudaSafeCall(cudaStreamDestroy(stream2));
		    }
		}

		~CUDAsolver() {
			int deviceCount = util::getNumDevice();
		    // multi-GPU support here, one thread = one GPU
		    #pragma omp parallel num_threads(deviceCount)
		    {
		    	// get device id = thread id
		    	int device_id_ = omp_get_thread_num();

		    	/*******************************/
				/* deallocating device Buffers */
				/*******************************/

				// deallocate memory
				cuda::cudaSafeCall(cudaFree(dev_vec_Buffer[device_id_]));
				cuda::cudaSafeCall(cudaFree(dev_mat_Buffer[device_id_]));
				cuda::cudaSafeCall(cudaFree(dev_pivotArray[device_id_]));
				cuda::cudaSafeCall(cudaFree(dev_InfoArray[device_id_]));
				cuda::cudaSafeCall(cudaFree(dev_mat_ptr[device_id_]));
				cuda::cudaSafeCall(cudaFree(dev_vec_ptr[device_id_]));
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
			std::cout << "\tsolving for " << n_solve << " particles on device " << constants::device_begin << "-" << constants::device_end << "\n";
#endif

		    // multi-GPU support here
		    #pragma omp parallel
		    {
		    	// thread id
		    	const int num_threads = omp_get_num_threads();
		    	const int thread_id = omp_get_thread_num();

		    	// get device id
		    	const int deviceCount = util::getNumDevice();
		    	const int device_id_  = deviceCount*thread_id/num_threads;
		    	const int device_id   = constants::device_begin + device_id_;

		    	// get solving range for device
		    	const size_t device_begin =       device_id_*n_solve/deviceCount;
		    	const size_t device_end   = (device_id_ + 1)*n_solve/deviceCount;
		    	const size_t device_size  = device_end - device_begin;
		    	// get thread range for device
		    	const size_t device_thread_begin =       device_id_*num_threads/deviceCount;
		    	const size_t device_thread_end   = (device_id_ + 1)*num_threads/deviceCount;
		    	const size_t device_num_thread   = device_thread_end - device_thread_begin;
		    	const size_t device_thread_id    =         thread_id - device_thread_begin;

		    	// get range for solving
		    	size_t solve_begin = device_begin +       device_thread_id*device_size/device_num_thread;
		    	size_t solve_end   = device_begin + (device_thread_id + 1)*device_size/device_num_thread;
		    	size_t solve_size  = solve_end - solve_begin;

		    	// actually solve
		    	solve_on_device(solve_begin, solve_size, device_id);
		    }
		}



		/// retrieve results
		/**
		 * TODO
		 */
		Float *get_res(size_t i) {
			return &vec_Buffer[dimension*i];
		}
	};
#endif



	/*******************************************************/
	/* dummy CPU solver while awaiting a CUDA batch solver */
	/*******************************************************/

	/// dummy batch CPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class CPUsolver {
	private:
		int dimension;
		size_t size;

		// Buffers
		std::vector<Float> vec_Buffer;
		std::vector<Float> mat_Buffer;
	
	public:
		/// allocate buffers for batch solver
		/**
		 * TODO
		 */
		CPUsolver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {
			vec_Buffer.resize(          dimension*size);
			mat_Buffer.resize(dimension*dimension*size);
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
		 * TODO
		 */
		void solve(size_t n_solve) {
#ifdef CUDA_DEBUG
		    /* debug: */
			std::cout << "\tsolving for " << n_solve << " particles on CPU\n";
#endif

			#pragma omp parallel
			{
				eigen::Vector<Float> RHS(dimension), res(dimension);
				eigen::Matrix<Float> Mp(dimension, dimension);

				#pragma omp for schedule(dynamic)
				for (size_t i = 0; i < n_solve; ++i) {
					// copy system to buffer
					auto [Mp_ref, RHS_ref] = get_system_reference(i);
					for (auto j = 0; j < dimension; ++j) {
						RHS[j] = RHS_ref[j];
						for (int k = 0; k < dimension; ++k)
							Mp(j, k) = Mp_ref[j + dimension*k];
					}

					// solve
					eigen::solve(Mp.data(), RHS.data(), res.data(), dimension);

					// save back to buffer
					Float *res_ref = get_res(i);
					for (int j = 0; j < dimension; ++j)
						res_ref[j] = res[j];
				}
			}
			
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
