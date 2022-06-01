#include "eigen.hpp"

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
		int dimension;
	
	public:
		/// allocate buffers for batch solver
		batch_solver(int dimension_) : dimension(dimension_) {
			/* TODO */
		}

		/// insert system into batch solver
		void insert_system(size_t i, const Float* M, const Float *RHS) {
			/* TODO */

			//throw std::runtime_error("CUDA batched solver \"insert_system\" not yet implemented !");
		}

		/// solve systems
		void solve() {
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