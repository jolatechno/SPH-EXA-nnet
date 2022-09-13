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
 * @brief "Eigen" (linear algebra) custom utility functions.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include "../CUDA/cuda.inl"
#if COMPILE_DEVICE
	#include "../CUDA/cuda-util.hpp"
#endif

#include <vector>
#include <cmath>
#include <tuple>

// base implementations
namespace eigen {
#if COMPILE_DEVICE
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
		template<class Other>
		fixed_size_matrix(Other const &other) : fixed_size_matrix() {
			*this = other;
		}
		template<class Other>
		fixed_size_matrix &operator=(Other const &other) {
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < m; ++j)
					operator()(i, j) = other[i][j];
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

		Type *data() {
			return weights.data();
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
		template<class Other>
		fixed_size_array(Other const &other) : fixed_size_array() {
			*this = other;
		}
		template<class Other>
		fixed_size_array &operator=(Other const &other) {
			for (int i = 0; i < n; ++i)
				operator[](i) = other[i];
		}

		Type inline &operator[](int i) {
			return weights[i];
		}
		const Type inline operator[](int i) const {
			return weights[i];
		}

		Type *data() {
			return weights.data();
		}
		const Type *data() const {
			return weights.data();
		}
	};

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
	HOST_DEVICE_FUN double dot(it1 const X_begin, it1 const X_end, it2 const Y_begin) {
		double res = 0;
		const int n = std::distance(X_begin, X_end);

		for (int i = 0; i < n; ++i)
			res += X_begin[i]*Y_begin[i];

		return res;
	}


	/// custom analytical solver
	template<typename Float>
	HOST_DEVICE_FUN void solve(Float *M, Float *RHS, Float *X, const int n, Float epsilon=0) {
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