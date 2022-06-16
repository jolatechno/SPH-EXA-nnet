#pragma once

#include <vector>
#include <cmath>
#include <tuple>

#ifdef USE_CUDA
	#include <cuda_runtime.h>
#endif
#include "../CUDA/cuda.inl"

// base implementations
namespace eigen {
#ifdef USE_CUDA
	// forward declarations
	namespace cuda {
		template<typename Type, int n, int m>
		class fixed_size_matrix;
		template<typename Type, int n>
		class fixed_size_array;
	}
#endif

	/// custom fixed-size matrix type
	template<typename Type, int n, int m>
	class fixed_size_matrix {
	private:
		std::vector<Type> weights;

	public:
		fixed_size_matrix() {
			weights.resize(n*m);
		}

		Type inline *operator[](int i) {
			return data() + i*m;
		}
		const Type inline *operator[](int i) const {
			return data() + i*m;
		}

		Type inline &operator()(int i, int j) {
			return weights[i*m + j];
		}
		Type inline operator()(int i, int j) const {
			return weights[i*m + j];
		}

		const Type *data() const {
			return weights.data();
		}
	};

	/// custom fixed-size matrix type
	template<typename Type, int n>
	class fixed_size_array {
	private:
		std::vector<Type> weights;

	public:
		fixed_size_array() {
			weights.resize(n);
		}

		Type inline &operator[](int i) {
			return weights[i];
		}
		const Type inline operator[](int i) const {
			return weights[i];
		}

		const Type *data() const {
			return weights.data();
		}
	};


#ifdef USE_CUDA
	namespace cuda {
		/// custom fixed-size matrix type
		template<typename Type, int n, int m>
		class fixed_size_matrix {
		private:
			Type *weights, *dev_weights;

		public:
			__host__ fixed_size_matrix() {
				/* TODO */
			}

			template<class Other>
			__host__ fixed_size_matrix(Other const &other) : fixed_size_matrix() {
				*this = other;
			}

			template<class Other>
			__host__ fixed_size_matrix &operator=(Other const &other) {
				/* TODO */
			}

			CUDA_FUNCTION_DECORATOR Type inline *operator[](int i) {
				return data() + i*m;
			}
			CUDA_FUNCTION_DECORATOR const Type inline *operator[](int i) const {
				return data() + i*m;
			}

			CUDA_FUNCTION_DECORATOR Type inline &operator()(int i, int j) {
				return CUDA_ACCESS(weights)[i*m + j];
			}
			CUDA_FUNCTION_DECORATOR Type inline operator()(int i, int j) const {
				return CUDA_ACCESS(weights)[i*m + j];
			}

			CUDA_FUNCTION_DECORATOR const Type *data() const {
				return CUDA_ACCESS(weights);
			}
		};

		/// custom fixed-size matrix type
		template<typename Type, int n>
		class fixed_size_array {
		private:
			Type *weights, *dev_weights;

		public:
			__host__ fixed_size_array() {
				/* TODO */
			}

			template<class Other>
			__host__ fixed_size_array(Other const &other) : fixed_size_array() {
				*this = other;
			}

			template<class Other>
			__host__ fixed_size_array &operator=(Other const &other) {
				/* TODO */
			}

			CUDA_FUNCTION_DECORATOR Type inline &operator[](int i) {
				return CUDA_ACCESS(weights)[i];
			}
			CUDA_FUNCTION_DECORATOR Type inline operator[](int i) const {
				return CUDA_ACCESS(weights)[i];
			}

			CUDA_FUNCTION_DECORATOR const Type *data() const {
				return CUDA_ACCESS(weights);
			}
		};
	}
#endif


	/// vector type
	template<typename Type>
	using Vector = std::vector<Type>;


	/// custom matrix type
	template<typename Type>
	class Matrix {
	private:
		std::vector<Type> weights;

	public:
		int n, m;

		Matrix() {}
		Matrix(int n_, int m_) {
			resize(n_, m_);
		}

		void resize(int n_, int m_) {
			n = n_;
			m = m_;
			weights.resize(n*m, 0);
		}

		Type inline &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type inline operator()(int i, int j) const {
			return weights[i + j*n];
		}

		Type inline *data() {
			return weights.data();
		}
	};


	/// dot product function
	template<class it1, class it2>
	CUDA_FUNCTION_DECORATOR double dot(it1 const X_begin, it1 const X_end, it2 const Y_begin) {
		double res = 0;
		const int n = std::distance(X_begin, X_end);

		for (int i = 0; i < n; ++i)
			res += X_begin[i]*Y_begin[i];

		return res;
	}


	/// custom analytical solver
	template<typename Float=double>
	CUDA_FUNCTION_DECORATOR void solve(Float *M, Float *RHS, Float *X, const int n, Float epsilon=0) {
		// reduce into upper triangular
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				// include the jth line
				Float weight = M[i + n*j];
				M[i + n*j] = 0;

				// eliminate
				if (std::abs(weight) > epsilon) {
					RHS[i] -= weight*RHS[j];
					for (int k = j + 1; k < n; ++k)
						M[i + n*k] -= weight*M[j + n*k];
				}
			}

			Float diagonal = M[i + n*i];
			M[i + n*i] = 1;

			// normalize
			RHS[i] /= diagonal;
			for (int j = i + 1; j < n; ++j)
				M[i + n*j] /= diagonal;
		}

		// "back propagate" to solve
		for (int i = n - 1; i >= 0; --i) {
			Float res = RHS[i];

			for (int j = i + 1; j < n; ++j) {
				Float weight = M[i + n*j];

				// if (std::abs(weight) > epsilon) // probably slower
				res -= weight*X[j];
			}

			X[i] = res;
		}
	}
}