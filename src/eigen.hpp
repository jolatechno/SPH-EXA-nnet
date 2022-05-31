#pragma once

#include <vector>
#include <cmath>
#include <tuple>

namespace eigen {
	/// custom matrix type
	template<typename Type>
	class matrix {
	private:
		std::vector<Type> weights;

	public:
		int n, m;

		matrix(int n_, int m_) : n(n_), m(m_) {
			weights.resize(n*m, 0);
		}

		Type inline &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type inline operator()(int i, int j) const {
			return weights[i + j*n];
		}
	};



	/// custom fixed-size matrix type
	template<typename Type, int n, int m>
	class fixed_size_matrix {
	private:
		std::vector<Type> weights;

	public:
		fixed_size_matrix() {
			weights.resize(n*m, 0);
		}

		Type inline &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type inline operator()(int i, int j) const {
			return weights[i + j*n];
		}
	};



	/// dot product function
	template<class Vector1, class Vector2>
	double dot(Vector1 const &X, Vector2 const &Y) {
		double res = 0;
		const int dimension = std::min(X.size(), Y.size());

		for (int i = 0; i < dimension; ++i)
			res += X[i]*Y[i];

		return res;
	}



	/// custom analytical solver
	template<typename Float, class Vector>
	Vector solve_gauss(matrix<Float> M, Vector RHS, Float epsilon=0) {
		if (M.n != M.m)
			throw std::runtime_error("can't use gaussian elimination on non-square matrices !");

		const int n = RHS.size();
		Vector X(n);

		// reduce into upper triangular
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				// include the jth line
				Float weight = M(i, j);
				M(i, j) = 0;

				// eliminate
				if (std::abs(weight) > epsilon) {
					RHS[i] -= weight*RHS[j];
					for (int k = j + 1; k < n; ++k)
						M(i, k) -= weight*M(j, k);
				}
			}

			Float diagonal = M(i, i);
			M(i, i) = 1;

			// normalize
			RHS[i] /= diagonal;
			for (int j = i + 1; j < n; ++j)
				M(i, j) /= diagonal;
		}

		// "back propagate" to solve
		for (int i = n - 1; i >= 0; --i) {
			Float res = RHS[i];

			for (int j = i + 1; j < n; ++j) {
				Float weight = M(i, j);

				// if (std::abs(weight) > epsilon) // probably slower
				res -= weight*X[j];
			}

			X[i] = res;
		}

		return X;
	}

	/// decompose a square matrix via LU decomposition
	template<typename Float>
	std::tuple<matrix<Float>, matrix<Float>&> LU_decompose(matrix<Float> M, Float epsilon=0) {
		if (M.n != M.m)
			throw std::runtime_error("can't LU-deompose non-square matrices !");

		matrix<Float> L(M.n, M.m);

		/* TODO */

		return {L, M};
	}

	/// custom analytical solver
	template<typename Float, class Vector>
	Vector solve_LU(matrix<Float> M, Vector RHS, Float epsilon=0) {
		// LU decompose
		auto [L, U] = LU_decompose(M, epsilon);

		// solve L*y = RHS
		/* TODO */

		// solve U*x = y (y = solution)
		/* TODO */

		return RHS;
	}
}