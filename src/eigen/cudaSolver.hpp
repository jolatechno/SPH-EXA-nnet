#include "eigen.hpp"

#ifdef CUDA_IMPLEMENTED
#include <cuda_runtime.h>

namespace eigen::cudasolver {
	/// batch size for the GPU batched solver
	size_t batch_size = 1000000;

	/// batch GPU solver
	/**
	 * TODO
	 */
	template<typename Float>
	class batch_solver {
	private:
		size_t size;
		int dimension;
	
	public:
		/// allocate buffers for batch solver
		batch_solver(size_t size_, int dimension_) : dimension(dimension_), size(size_) {

			int deviceCount;
		    cudaError_t e = cudaGetDeviceCount(&deviceCount);
		    std::cout << "(batch_solver) number of cuda device:" << (e == cudaSuccess ? deviceCount : -1) << "\n";

			/* TODO */
		}

		/// insert system into batch solver
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			/* TODO */

			//throw std::runtime_error("CUDA batched solver \"insert_system\" not yet implemented !");
		}

		/// solve systems
		void solve(size_t n_solve) {
			/* TODO */

			throw std::runtime_error("CUDA batched solver \"solve\" not yet implemented !");
		}

		/// retrieve results
		void get_res(size_t i, Float *res) {
			/* TODO */

			throw std::runtime_error("CUDA batched solver \"get_res\" not yet implemented !");
		}
	};
}



#else

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dummy CPU solver while awaiting a CUDA batch solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

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
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			for (int j = 0; j < dimension; ++j) {
				RHS_Buffer[i][j] = RHS[j];
				for (int k = 0; k < dimension; ++k)
					mat_Buffer[i](j, k) = M[j + dimension*k];
			}
		}

		/// solve systems
		void solve(size_t n_solve) {
			#pragma omp parallel for schedule(dynamic)
			for (size_t i = 0; i < n_solve; ++i)
				res_Buffer[i] = eigen::solve(mat_Buffer[i], RHS_Buffer[i]);
		}

		/// retrieve results
		void get_res(size_t i, Float *res) {
			for (int j = 0; j < dimension; ++j)
				res[j] = res_Buffer[i][j];
		}
	};
}
#endif